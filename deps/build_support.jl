module BuildSupport
using CMake_jll, Eigen_jll, Scratch, SHA, Libdl
const PACKAGE_UUID = Base.UUID("3ead10a5-a636-4c1d-b4eb-bf43a9e33d59")
const ROOT = normpath(joinpath(@__DIR__, ".."))
const BUILD_LOCK = ReentrantLock()

function library_name()
    (Sys.iswindows() ? "spectralmm." : "libspectralmm.") * Libdl.dlext
end

function cache_directory()
    files = [joinpath(ROOT, "cpp", "CMakeLists.txt"), @__FILE__]
    for (dir, _, names) in walkdir(joinpath(ROOT, "src", "native"))
        append!(files, joinpath.(dir, sort(names)))
    end
    buffer = IOBuffer()
    print(buffer, Sys.MACHINE, VERSION.major, '.', VERSION.minor, Eigen_jll.artifact_dir)
    for key in ("CC", "CXX", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "MACOSX_DEPLOYMENT_TARGET", "SPECTRALMM_BUILD_BLAS")
        print(buffer, key, '=', get(ENV, key, ""), '\0')
    end
    for file in sort(files)
        print(buffer, relpath(file, ROOT), '\0')
        write(buffer, read(file))
    end
    key = "core-" * bytes2hex(sha256(take!(buffer)))[1:24]
    get_scratch!(PACKAGE_UUID, key)
end

function ensure_library()
    if haskey(ENV, "SPECTRALMM_LIBRARY")
        path = abspath(ENV["SPECTRALMM_LIBRARY"])
        isfile(path) || error("SPECTRALMM_LIBRARY does not exist: $path")
        return path
    end
    lock(BUILD_LOCK) do
        cache = cache_directory()
        destination = joinpath(cache, library_name())
        isfile(destination) && return destination
        @info "Building SpectralMM C++ library (cached for subsequent sessions)"
        eigen = joinpath(Eigen_jll.artifact_dir, "include", "eigen3")
        isfile(joinpath(eigen, "Eigen", "Core")) || error("Eigen headers missing from the package artifact")
        mktempdir(cache) do work
            build = joinpath(work, "build")
            stage = joinpath(work, "install")
            cmake = CMake_jll.cmake()
            blas = get(ENV, "SPECTRALMM_BUILD_BLAS", "AUTO")
            blas in ("ON", "OFF", "AUTO") || error("SPECTRALMM_BUILD_BLAS must be ON, OFF or AUTO")
            # Julia exports its library search path to child processes. On Linux,
            # unconstrained discovery can select libblastrampoline, whose LP64
            # entry points are not necessarily configured by Julia's ILP64 BLAS.
            # Prefer a real LP64 OpenBLAS; AUTO falls back to Eigen if absent.
            blas_args = Sys.islinux() ? ["-DBLA_VENDOR=OpenBLAS", "-DBLA_SIZEOF_INTEGER=4"] : String[]
            run(`$cmake -S $(joinpath(ROOT,"cpp")) -B $build -DCMAKE_BUILD_TYPE=Release -DEIGEN3_INCLUDE_DIR=$eigen -DSPECTRALMM_USE_BLAS=$blas $blas_args -DCMAKE_INSTALL_PREFIX=$stage`)
            run(`$cmake --build $build --config Release --parallel 2`)
            run(`$cmake --install $build --config Release`)
            built = joinpath(stage, Sys.iswindows() ? "bin" : "lib", library_name())
            isfile(built) || error("SpectralMM build did not produce the expected library")
            # Separate processes build in private directories, then publish atomically.
            # A completed equivalent build can be reused without replacing a loaded DLL.
            if !isfile(destination)
                try
                    mv(built, destination)
                catch
                    isfile(destination) || rethrow()
                end
            end
        end
        destination
    end
end
end
