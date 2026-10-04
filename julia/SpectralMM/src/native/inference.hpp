// Post-fit inference. Never use the optimizer's low-rank preconditioner here.
Mat weighted_information(const Design& X, const Vec& w) {
    const auto n=X.x.n, p=X.x.p, d=X.raw.p;
    Mat H=Mat::Zero(p,p);
    if (X.raw.storage) {
        Sparse weighted=X.sparse();
        for (int64_t j=0;j<weighted.outerSize();++j)
            for (Sparse::InnerIterator it(weighted,j);it;++it) it.valueRef()*=w[it.row()];
        H.bottomRightCorner(d,d)=Mat(X.sparse().transpose()*weighted);
        if (X.intercept) {
            Vec cross(p); X.tmv_into(w,cross);
            H.col(0)=cross; H.row(0)=cross.transpose();
        }
    } else {
        // Bounded row blocks avoid an additional n-by-p dense matrix.
        for (int64_t first=0;first<n;first+=256) {
            const auto count=std::min<int64_t>(256,n-first);
            Mat block(count,p);
            if (X.intercept) block.col(0).setOnes();
            block.rightCols(d)=X.dense().middleRows(first,count);
            H.noalias()+=block.transpose()*w.segment(first,count).asDiagonal()*block;
        }
    }
    return .5*(H+H.transpose()).eval();
}
