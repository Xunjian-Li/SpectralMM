# Changelog

## 0.1.0

Initial release of the Julia, R and Python statistical interfaces, with shared
C++ support and a pure Julia backend selected by default in Julia.

PseudoHuber now defaults to delta=1 across constructors, legacy string APIs,
Julia backends and the C ABI. Earlier development string/Julia defaults were
1.345; explicitly pass delta=1.345 to reproduce those fits. This intentional
parameter-default change does not alter the solver algorithms or tolerances.
