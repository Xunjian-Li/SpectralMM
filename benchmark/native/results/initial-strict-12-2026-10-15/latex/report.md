# Python/R benchmark results

Timing source: initial. Times are seconds. Gradient convergence is assessed using the common gradient thresholds; warnings remain separate.

## Python: dense, glm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| gaussian | Spectral-MM (PCG) | 0.013 | 25 | 9.375702441291822e-09 | Yes* |
| gaussian | Spectral-MM (MM) | 0.023 | 23 | 2.0396441461863895e-09 | Yes* |
| gaussian | LSMR | 0.009 | 2 | 7.383070604264562e-09 | Yes* |
| gaussian | CRLS | 0.009 | 2 | 7.383070601102773e-09 | Yes* |
| gaussian | CGLS | 0.009 | 2 | 7.342391059879092e-09 | Yes* |
| gaussian | CG | 0.009 | 2 | 7.342391100031604e-09 | Yes* |
| gaussian | LSQR | 0.009 | 2 | 7.34239103277943e-09 | Yes* |
| gaussian | GLM | 0.446 | 3 | 5.199337715910728e-15 | Yes |
| bernoulli | Spectral-MM (PCG) | 0.064 | 54 | 6.2129756534675995e-09 | Yes* |
| bernoulli | Spectral-MM (MM) | 0.077 | 16 | 6.54527218740102e-10 | Yes* |
| bernoulli | LSMR | 0.165 | 11 | 5.285817202196314e-09 | Yes* |
| bernoulli | CGLS | 0.166 | 11 | 5.257573321198929e-09 | Yes* |
| bernoulli | LSQR | 0.169 | 11 | 5.347934559863004e-09 | Yes* |
| bernoulli | CRLS | 0.176 | 11 | 5.2854190686246765e-09 | Yes* |
| bernoulli | CG | 0.179 | 11 | 5.310592979036845e-09 | Yes* |
| bernoulli | GLM | 3.308 | 42 | 5.077499706542551e-16 | Yes* |
| probit | Spectral-MM (PCG) | 0.062 | 37 | 7.558540315023123e-09 | Yes* |
| probit | Spectral-MM (MM) | 0.059 | 17 | 4.0827082484659205e-09 | Yes* |
| probit | LSMR | 0.164 | 11 | 2.944011593398957e-10 | Yes* |
| probit | CGLS | 0.165 | 11 | 2.834854594388923e-10 | Yes* |
| probit | LSQR | 0.167 | 11 | 2.8549095383332847e-10 | Yes* |
| probit | CRLS | 0.178 | 11 | 2.945294974352363e-10 | Yes* |
| probit | CG | 0.180 | 11 | 2.853484724375071e-10 | Yes* |
| probit | GLM | 3.077 | 39 | 2.5107153332649402e-15 | Yes* |
| poisson | Spectral-MM (PCG) | 0.031 | 35 | 5.547862072618161e-09 | Yes* |
| poisson | Spectral-MM (MM) | 0.055 | 21 | 2.384612559347594e-09 | Yes* |
| poisson | LSMR | 0.038 | 6 | 5.640579038557931e-10 | Yes* |
| poisson | CGLS | 0.040 | 6 | 6.102083091655024e-10 | Yes* |
| poisson | LSQR | 0.041 | 6 | 6.178331872974068e-10 | Yes* |
| poisson | CRLS | 0.041 | 6 | 5.663233018001569e-10 | Yes* |
| poisson | CG | 0.044 | 6 | 5.608338704945655e-10 | Yes* |
| poisson | GLM | 1.036 | 11 | 8.21116451271085e-15 | Yes |
| gamma | Spectral-MM (PCG) | 0.028 | 64 | 7.349935043055522e-09 | Yes* |
| gamma | Spectral-MM (MM) | 0.052 | 126 | 7.952974842666157e-09 | Yes* |
| gamma | CGLS | 0.189 | 45 | 7.499892211336158e-09 | Yes* |
| gamma | LSQR | 0.192 | 45 | 7.499892394293847e-09 | Yes* |
| gamma | LSMR | 0.194 | 49 | 5.407410684011779e-09 | Yes* |
| gamma | CRLS | 0.208 | 49 | 5.407410749811868e-09 | Yes* |
| gamma | CG | 0.226 | 50 | 5.336893436343755e-09 | Yes* |
| gamma | GLM | 73.910 | 1000 | 0.024089318742149066 | No |
| negative_binomial | Spectral-MM (PCG) | 0.025 | 42 | 7.159306248288705e-09 | Yes* |
| negative_binomial | Spectral-MM (MM) | 0.048 | 80 | 7.582824309521239e-09 | Yes* |
| negative_binomial | CRLS | 0.121 | 31 | 2.8907267525243685e-09 | Yes* |
| negative_binomial | CG | 0.126 | 28 | 4.982291086243617e-09 | Yes* |
| negative_binomial | LSMR | 0.129 | 31 | 2.890726521315119e-09 | Yes* |
| negative_binomial | LSQR | 0.133 | 28 | 4.982291201691728e-09 | Yes* |
| negative_binomial | CGLS | 0.134 | 29 | 6.20682142857854e-09 | Yes* |
| negative_binomial | GLM | 1.773 | 21 | 5.295520972658119e-09 | Yes |
## Python: dense, nonglm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| smooth_quantile | Spectral-MM (PCG) | 0.073 | 47 | 9.624861736684399e-09 | Yes* |
| smooth_quantile | Spectral-MM (MM) | 0.129 | 34 | 5.910945699414259e-09 | Yes* |
| smooth_quantile | CRLS | 0.284 | 20 | 7.622595968027201e-10 | Yes* |
| smooth_quantile | LSQR | 0.325 | 20 | 9.190754672859185e-10 | Yes* |
| smooth_quantile | LSMR | 0.325 | 22 | 2.413206456661271e-09 | Yes* |
| smooth_quantile | CGLS | 0.334 | 20 | 1.1148975113488357e-09 | Yes* |
| smooth_quantile | CG | 0.430 | 26 | 7.725690280513556e-10 | Yes* |
| expectile | Spectral-MM (PCG) | 0.018 | 29 | 7.062885957233001e-09 | Yes* |
| expectile | Spectral-MM (MM) | 0.028 | 22 | 2.8196279839541473e-09 | Yes* |
| expectile | LSMR | 0.021 | 5 | 3.4742954427843086e-10 | Yes* |
| expectile | CRLS | 0.021 | 5 | 3.4742960087359934e-10 | Yes* |
| expectile | CGLS | 0.022 | 5 | 3.588936620254238e-10 | Yes* |
| expectile | LSQR | 0.022 | 5 | 3.5889354253822004e-10 | Yes* |
| expectile | CG | 0.023 | 5 | 3.5889366118659173e-10 | Yes* |
| pseudo_huber | Spectral-MM (PCG) | 0.021 | 31 | 7.864111171628179e-09 | Yes* |
| pseudo_huber | Spectral-MM (MM) | 0.030 | 21 | 9.29939014833731e-09 | Yes* |
| pseudo_huber | LSMR | 0.044 | 9 | 3.7857156608031065e-09 | Yes* |
| pseudo_huber | CRLS | 0.045 | 9 | 3.785715642849402e-09 | Yes* |
| pseudo_huber | CGLS | 0.046 | 9 | 3.6912091917868533e-09 | Yes* |
| pseudo_huber | LSQR | 0.047 | 9 | 3.6911334720035173e-09 | Yes* |
| pseudo_huber | CG | 0.048 | 9 | 3.691133032082584e-09 | Yes* |
| student_t | Spectral-MM (PCG) | 0.026 | 48 | 6.5179233804670484e-09 | Yes* |
| student_t | Spectral-MM (MM) | 0.044 | 35 | 4.829366336772774e-09 | Yes* |
| student_t | LSMR | 0.090 | 25 | 7.136786799075061e-09 | Yes* |
| student_t | CRLS | 0.092 | 25 | 7.136786809510198e-09 | Yes* |
| student_t | LSQR | 0.093 | 24 | 8.53596990651754e-09 | Yes* |
| student_t | CGLS | 0.093 | 24 | 8.5359698115617e-09 | Yes* |
| student_t | CG | 0.096 | 24 | 8.535969876887477e-09 | Yes* |
## Python: sparse, glm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| gaussian | Spectral-MM (PCG) | 0.005 | 26 | 5.555690048930183e-09 | Yes* |
| gaussian | Spectral-MM (MM) | 0.007 | 21 | 6.64243469205578e-09 | Yes* |
| gaussian | CG | 0.002 | 2 | 6.7634972209351435e-09 | Yes* |
| gaussian | CRLS | 0.002 | 2 | 6.813425182655811e-09 | Yes* |
| gaussian | CGLS | 0.002 | 2 | 6.763497210540748e-09 | Yes* |
| gaussian | LSQR | 0.002 | 2 | 6.763497198113042e-09 | Yes* |
| gaussian | LSMR | 0.002 | 2 | 6.813425183440844e-09 | Yes* |
| gaussian | GLM | 0.458 | 3 | 1.3425738989946144e-14 | Yes |
| bernoulli | Spectral-MM (PCG) | 0.051 | 86 | 5.116558260543447e-09 | Yes* |
| bernoulli | Spectral-MM (MM) | 0.038 | 34 | 5.598066201103435e-09 | Yes* |
| bernoulli | CG | 0.088 | 16 | 2.381243481903339e-09 | Yes* |
| bernoulli | CRLS | 0.090 | 16 | 1.3507372444086147e-09 | Yes* |
| bernoulli | CGLS | 0.092 | 16 | 1.5723719297277605e-09 | Yes* |
| bernoulli | LSMR | 0.093 | 16 | 1.3482876019731289e-09 | Yes* |
| bernoulli | LSQR | 0.093 | 16 | 2.0894693669355433e-09 | Yes* |
| bernoulli | GLM | 3.682 | 47 | 1.4355540439487457e-15 | Yes* |
| probit | Spectral-MM (PCG) | 0.097 | 80 | 6.273381279736483e-09 | Yes* |
| probit | Spectral-MM (MM) | 0.042 | 25 | 2.088420216083075e-09 | Yes* |
| probit | CG | 0.084 | 14 | 8.006617670134431e-09 | Yes* |
| probit | CGLS | 0.086 | 14 | 8.076000881330314e-09 | Yes* |
| probit | LSQR | 0.089 | 14 | 7.937953993249215e-09 | Yes* |
| probit | CRLS | 0.090 | 16 | 5.0204946642271295e-09 | Yes* |
| probit | LSMR | 0.095 | 16 | 4.6911374919238804e-09 | Yes* |
| probit | GLM | 3.340 | 42 | 9.988118768066877e-15 | Yes* |
| poisson | Spectral-MM (PCG) | 0.008 | 30 | 5.853214628416954e-09 | Yes* |
| poisson | Spectral-MM (MM) | 0.010 | 26 | 9.747212753774009e-09 | Yes* |
| poisson | CRLS | 0.005 | 5 | 9.873729445428638e-09 | Yes* |
| poisson | CG | 0.005 | 5 | 9.914504725204766e-09 | Yes* |
| poisson | LSMR | 0.005 | 5 | 9.873726827384152e-09 | Yes* |
| poisson | CGLS | 0.005 | 5 | 9.914500168006878e-09 | Yes* |
| poisson | LSQR | 0.006 | 5 | 9.91444810844952e-09 | Yes* |
| poisson | GLM | 0.756 | 7 | 8.570630140618649e-15 | Yes |
| gamma | Spectral-MM (PCG) | 0.018 | 82 | 9.437090129709526e-09 | Yes* |
| gamma | Spectral-MM (MM) | 0.030 | 124 | 7.43717666871789e-09 | Yes* |
| gamma | CRLS | 0.033 | 40 | 9.431048214282669e-09 | Yes* |
| gamma | CG | 0.035 | 40 | 9.414925122720327e-09 | Yes* |
| gamma | CGLS | 0.036 | 40 | 9.414925127649183e-09 | Yes* |
| gamma | LSMR | 0.037 | 40 | 9.43104820550926e-09 | Yes* |
| gamma | LSQR | 0.037 | 40 | 9.41492513454592e-09 | Yes* |
| gamma | GLM | 75.367 | 1000 | 0.08252052540681926 | No |
| negative_binomial | Spectral-MM (PCG) | 0.010 | 36 | 5.5641150103192465e-09 | Yes* |
| negative_binomial | Spectral-MM (MM) | 0.017 | 58 | 6.930174621941299e-09 | Yes* |
| negative_binomial | CRLS | 0.016 | 17 | 6.893521554237646e-09 | Yes* |
| negative_binomial | CG | 0.016 | 17 | 6.157102602764757e-09 | Yes* |
| negative_binomial | CGLS | 0.017 | 17 | 6.157102552741995e-09 | Yes* |
| negative_binomial | LSMR | 0.017 | 17 | 6.893521610089283e-09 | Yes* |
| negative_binomial | LSQR | 0.017 | 17 | 6.157102569553342e-09 | Yes* |
| negative_binomial | GLM | 1.435 | 16 | 1.1645474241980602e-08 | No |
## Python: sparse, nonglm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| smooth_quantile | Spectral-MM (PCG) | 0.020 | 42 | 8.929861300924642e-09 | Yes* |
| smooth_quantile | Spectral-MM (MM) | 0.066 | 34 | 7.312701626991242e-09 | Yes* |
| smooth_quantile | CRLS | 0.071 | 20 | 3.16942456135806e-10 | Yes* |
| smooth_quantile | LSMR | 0.072 | 19 | 1.0968372259815305e-09 | Yes* |
| smooth_quantile | CG | 0.077 | 19 | 5.990160209700238e-09 | Yes* |
| smooth_quantile | CGLS | 0.077 | 19 | 9.549610413146508e-09 | Yes* |
| smooth_quantile | LSQR | 0.080 | 19 | 5.665986517759101e-09 | Yes* |
| expectile | Spectral-MM (PCG) | 0.006 | 25 | 9.932402066639862e-09 | Yes* |
| expectile | Spectral-MM (MM) | 0.010 | 33 | 9.584616785679177e-09 | Yes* |
| expectile | CRLS | 0.005 | 5 | 1.3204226942424644e-10 | Yes* |
| expectile | CG | 0.005 | 5 | 1.112056799838097e-10 | Yes* |
| expectile | LSMR | 0.005 | 5 | 1.3204226778348434e-10 | Yes* |
| expectile | CGLS | 0.005 | 5 | 1.1120565124722514e-10 | Yes* |
| expectile | LSQR | 0.005 | 5 | 1.1120566213598631e-10 | Yes* |
| pseudo_huber | Spectral-MM (PCG) | 0.008 | 26 | 6.165290032182725e-09 | Yes* |
| pseudo_huber | Spectral-MM (MM) | 0.013 | 40 | 8.763296059198596e-09 | Yes* |
| pseudo_huber | CRLS | 0.012 | 10 | 4.0681589673171045e-09 | Yes* |
| pseudo_huber | CG | 0.012 | 10 | 4.042728956371085e-09 | Yes* |
| pseudo_huber | LSMR | 0.012 | 10 | 4.0681589784853936e-09 | Yes* |
| pseudo_huber | CGLS | 0.013 | 10 | 4.0427289475377205e-09 | Yes* |
| pseudo_huber | LSQR | 0.013 | 10 | 4.0427289959738066e-09 | Yes* |
| student_t | Spectral-MM (PCG) | 0.010 | 42 | 8.822866762450349e-09 | Yes* |
| student_t | Spectral-MM (MM) | 0.015 | 35 | 6.266518957873187e-09 | Yes* |
| student_t | CRLS | 0.022 | 25 | 6.22869254010707e-09 | Yes* |
| student_t | CG | 0.023 | 25 | 6.218364531607168e-09 | Yes* |
| student_t | LSMR | 0.023 | 25 | 6.2286925433009086e-09 | Yes* |
| student_t | CGLS | 0.023 | 25 | 6.2183645516390834e-09 | Yes* |
| student_t | LSQR | 0.024 | 25 | 7.019269657535992e-09 | Yes* |
## R: dense, glm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| gaussian | Spectral-MM (PCG) | 0.041 | 25 | 9.37570248657204e-09 | Yes |
| gaussian | Spectral-MM (MM) | 0.073 | 23 | 2.03964419297306e-09 | Yes |
| gaussian | LSMR | 0.026 | 2 | 7.38307057883471e-09 | Yes |
| gaussian | CRLS | 0.027 | 2 | 7.38307062690122e-09 | Yes |
| gaussian | CG | 0.028 | 2 | 7.34239106635191e-09 | Yes |
| gaussian | LSQR | 0.028 | 2 | 7.3423910324928e-09 | Yes |
| gaussian | CGLS | 0.029 | 2 | 7.34239106109785e-09 | Yes |
| gaussian | GLM | 0.501 | 2 | 1.83377158591344e-15 | Yes |
| bernoulli | Spectral-MM (PCG) | 0.154 | 54 | 6.21297558142873e-09 | Yes |
| bernoulli | Spectral-MM (MM) | 0.208 | 16 | 6.54527218740255e-10 | Yes |
| bernoulli | LSMR | 0.497 | 11 | 5.28782840480791e-09 | Yes |
| bernoulli | CGLS | 0.500 | 11 | 5.24138234118814e-09 | Yes |
| bernoulli | CG | 0.500 | 11 | 5.34159631610364e-09 | Yes |
| bernoulli | CRLS | 0.500 | 11 | 5.28669769091171e-09 | Yes |
| bernoulli | LSQR | 0.505 | 11 | 5.2905171463376e-09 | Yes |
| bernoulli | GLM | 15.075 | 60 | 4.03458524897639e-14 | Yes* |
| probit | Spectral-MM (PCG) | 0.134 | 37 | 7.5585403145348e-09 | Yes |
| probit | Spectral-MM (MM) | 0.161 | 17 | 4.08270824846601e-09 | Yes |
| probit | CRLS | 0.518 | 11 | 2.94535899435706e-10 | Yes |
| probit | LSMR | 0.519 | 11 | 2.94628530592675e-10 | Yes |
| probit | CGLS | 0.527 | 11 | 2.83049401334184e-10 | Yes |
| probit | CG | 0.528 | 11 | 2.84378404838162e-10 | Yes |
| probit | LSQR | 0.528 | 11 | 2.83181398815107e-10 | Yes |
| probit | GLM | 10.026 | 39 | 9.6447409429685e-17 | Yes* |
| poisson | Spectral-MM (PCG) | 0.084 | 34 | 6.4639435799887e-09 | Yes |
| poisson | Spectral-MM (MM) | 0.157 | 21 | 2.38461233055528e-09 | Yes |
| poisson | CRLS | 0.107 | 6 | 5.67354992525951e-10 | Yes |
| poisson | LSMR | 0.108 | 6 | 5.65275530049142e-10 | Yes |
| poisson | CGLS | 0.117 | 6 | 6.08437593560741e-10 | Yes |
| poisson | CG | 0.117 | 6 | 5.56076626384097e-10 | Yes |
| poisson | LSQR | 0.117 | 6 | 5.42057634550849e-10 | Yes |
| poisson | GLM | 2.777 | 11 | 2.37317112765775e-15 | Yes |
| gamma | Spectral-MM (PCG) | 0.079 | 64 | 7.34993507452802e-09 | Yes |
| gamma | Spectral-MM (MM) | 0.131 | 126 | 7.95297489930504e-09 | Yes |
| gamma | CG | 0.561 | 45 | 7.49989233337744e-09 | Yes |
| gamma | LSMR | 0.568 | 49 | 5.40741054094387e-09 | Yes |
| gamma | CGLS | 0.573 | 45 | 7.49989239247515e-09 | Yes |
| gamma | CRLS | 0.583 | 49 | 5.40741066321515e-09 | Yes |
| gamma | LSQR | 0.586 | 45 | 7.49989225118453e-09 | Yes |
| gamma | GLM | 255.080 | 1000 | 0.02408931874215 | No* |
| negative_binomial | Spectral-MM (PCG) | 0.069 | 42 | 7.15930629218411e-09 | Yes |
| negative_binomial | Spectral-MM (MM) | 0.125 | 80 | 7.58282427925083e-09 | Yes |
| negative_binomial | CRLS | 0.349 | 28 | 7.3406404633933e-09 | Yes |
| negative_binomial | LSMR | 0.352 | 28 | 7.34064036885908e-09 | Yes |
| negative_binomial | CG | 0.388 | 28 | 4.98229107538062e-09 | Yes |
| negative_binomial | CGLS | 0.391 | 28 | 4.98229097454318e-09 | Yes |
| negative_binomial | LSQR | 0.391 | 28 | 4.98229107920095e-09 | Yes |
| negative_binomial | GLM | 4.256 | 17 | 1.00463707377353e-07 | No |
## R: dense, nonglm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| smooth_quantile | Spectral-MM (PCG) | 0.201 | 49 | 7.77957548208421e-09 | Yes |
| smooth_quantile | Spectral-MM (MM) | 0.350 | 34 | 5.91094582033205e-09 | Yes |
| smooth_quantile | CRLS | 0.871 | 21 | 7.89455104806784e-10 | Yes |
| smooth_quantile | CG | 0.895 | 19 | 8.77548544644564e-10 | Yes |
| smooth_quantile | LSMR | 0.968 | 24 | 8.11350155939743e-10 | Yes |
| smooth_quantile | LSQR | 1.226 | 26 | 7.59341136521995e-10 | Yes |
| smooth_quantile | CGLS | 1.228 | 26 | 8.04003342820884e-10 | Yes |
| expectile | Spectral-MM (PCG) | 0.048 | 29 | 7.06288575905192e-09 | Yes |
| expectile | Spectral-MM (MM) | 0.083 | 22 | 2.81962797229924e-09 | Yes |
| expectile | CRLS | 0.064 | 5 | 3.47429587267852e-10 | Yes |
| expectile | LSMR | 0.065 | 5 | 3.474295865251e-10 | Yes |
| expectile | CGLS | 0.069 | 5 | 3.58893635770886e-10 | Yes |
| expectile | CG | 0.069 | 5 | 3.58893566480707e-10 | Yes |
| expectile | LSQR | 0.069 | 5 | 3.58893514118681e-10 | Yes |
| pseudo_huber | Spectral-MM (PCG) | 0.054 | 30 | 7.54965802160311e-09 | Yes |
| pseudo_huber | Spectral-MM (MM) | 0.085 | 21 | 9.29939017063728e-09 | Yes |
| pseudo_huber | CRLS | 0.136 | 9 | 3.7857156886224e-09 | Yes |
| pseudo_huber | LSMR | 0.137 | 9 | 3.78571542911007e-09 | Yes |
| pseudo_huber | LSQR | 0.147 | 9 | 3.69113202462219e-09 | Yes |
| pseudo_huber | CG | 0.147 | 9 | 3.69113301536317e-09 | Yes |
| pseudo_huber | CGLS | 0.148 | 9 | 3.69119395351419e-09 | Yes |
| student_t | Spectral-MM (PCG) | 0.065 | 48 | 6.66020766060982e-09 | Yes |
| student_t | Spectral-MM (MM) | 0.126 | 35 | 4.82936636852485e-09 | Yes |
| student_t | CRLS | 0.273 | 25 | 7.13678681179952e-09 | Yes |
| student_t | LSMR | 0.276 | 25 | 7.1367868621012e-09 | Yes |
| student_t | CG | 0.285 | 24 | 8.53596982451171e-09 | Yes |
| student_t | CGLS | 0.288 | 24 | 8.53596989014727e-09 | Yes |
| student_t | LSQR | 0.288 | 24 | 8.53596989235939e-09 | Yes |
## R: sparse, glm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| gaussian | Spectral-MM (PCG) | 0.005 | 26 | 5.55569004591302e-09 | Yes |
| gaussian | Spectral-MM (MM) | 0.007 | 21 | 6.64243468352561e-09 | Yes |
| gaussian | CRLS | 0.002 | 2 | 6.81342518641713e-09 | Yes |
| gaussian | CG | 0.002 | 2 | 6.76349715648232e-09 | Yes |
| gaussian | LSMR | 0.002 | 2 | 6.81342517386815e-09 | Yes |
| gaussian | LSQR | 0.002 | 2 | 6.76349716929732e-09 | Yes |
| gaussian | CGLS | 0.003 | 2 | 6.76349716802104e-09 | Yes |
| gaussian | GLM | 0.480 | 2 | 1.01847605850917e-14 | Yes |
| bernoulli | Spectral-MM (PCG) | 0.053 | 86 | 5.1165397077151e-09 | Yes |
| bernoulli | Spectral-MM (MM) | 0.037 | 34 | 5.598066201802e-09 | Yes |
| bernoulli | CRLS | 0.100 | 16 | 1.3385773752323e-09 | Yes |
| bernoulli | CGLS | 0.101 | 16 | 2.00306202023728e-09 | Yes |
| bernoulli | CG | 0.101 | 16 | 2.59369636677606e-09 | Yes |
| bernoulli | LSMR | 0.103 | 16 | 1.34316791636139e-09 | Yes |
| bernoulli | LSQR | 0.103 | 16 | 2.93087142872683e-09 | Yes |
| bernoulli | GLM | 24.826 | 98 | 1.05323031174222e-13 | Yes* |
| probit | Spectral-MM (PCG) | 0.092 | 73 | 6.81714215479675e-09 | Yes |
| probit | Spectral-MM (MM) | 0.042 | 25 | 2.08842021608235e-09 | Yes |
| probit | CG | 0.093 | 14 | 8.10135325200169e-09 | Yes |
| probit | CGLS | 0.095 | 14 | 8.22158615883129e-09 | Yes |
| probit | LSQR | 0.096 | 14 | 8.12181215318907e-09 | Yes |
| probit | CRLS | 0.101 | 16 | 5.14260255585416e-09 | Yes |
| probit | LSMR | 0.104 | 16 | 4.42408532807398e-09 | Yes |
| probit | GLM | 10.128 | 40 | 1.38749498198645e-16 | Yes* |
| poisson | Spectral-MM (PCG) | 0.008 | 30 | 5.85321462289198e-09 | Yes |
| poisson | Spectral-MM (MM) | 0.010 | 26 | 9.74721281570527e-09 | Yes |
| poisson | CRLS | 0.006 | 5 | 9.87371360314685e-09 | Yes |
| poisson | CG | 0.006 | 5 | 9.91442981006455e-09 | Yes |
| poisson | LSMR | 0.006 | 5 | 9.87372491008423e-09 | Yes |
| poisson | CGLS | 0.006 | 5 | 9.91438934607227e-09 | Yes |
| poisson | LSQR | 0.006 | 5 | 9.91438613535125e-09 | Yes |
| poisson | GLM | 1.520 | 6 | 3.35158136184677e-13 | Yes |
| gamma | Spectral-MM (PCG) | 0.017 | 82 | 9.43709017254856e-09 | Yes |
| gamma | Spectral-MM (MM) | 0.030 | 124 | 7.43717659713208e-09 | Yes |
| gamma | CRLS | 0.036 | 40 | 9.43104820717714e-09 | Yes |
| gamma | CG | 0.038 | 40 | 9.41492510624382e-09 | Yes |
| gamma | LSMR | 0.039 | 40 | 9.43104820637887e-09 | Yes |
| gamma | CGLS | 0.040 | 41 | 5.05412099367218e-09 | Yes |
| gamma | LSQR | 0.040 | 40 | 9.41492512560739e-09 | Yes |
| gamma | GLM | 250.943 | 1000 | 0.0825205254068171 | No* |
| negative_binomial | Spectral-MM (PCG) | 0.010 | 36 | 5.56411497455613e-09 | Yes |
| negative_binomial | Spectral-MM (MM) | 0.017 | 58 | 6.93017462220187e-09 | Yes |
| negative_binomial | CRLS | 0.017 | 17 | 6.89352153698016e-09 | Yes |
| negative_binomial | CG | 0.018 | 17 | 6.1571025187106e-09 | Yes |
| negative_binomial | LSMR | 0.018 | 17 | 6.89352161408633e-09 | Yes |
| negative_binomial | CGLS | 0.018 | 17 | 6.15710257167059e-09 | Yes |
| negative_binomial | LSQR | 0.019 | 17 | 6.1571025221701e-09 | Yes |
| negative_binomial | GLM | 3.306 | 13 | 2.05682281621985e-07 | No |
## R: sparse, nonglm

| Model | Method | Time (s) | Iteration | Rel. Grad. | Gradient converged |
|---|---|---:|---:|---:|---|
| smooth_quantile | Spectral-MM (PCG) | 0.020 | 43 | 6.0725320661394e-09 | Yes |
| smooth_quantile | Spectral-MM (MM) | 0.064 | 34 | 7.31270176976842e-09 | Yes |
| smooth_quantile | CRLS | 0.076 | 19 | 1.82383732906617e-09 | Yes |
| smooth_quantile | LSMR | 0.079 | 19 | 1.15271322318977e-09 | Yes |
| smooth_quantile | CG | 0.086 | 19 | 3.09981025479142e-09 | Yes |
| smooth_quantile | CGLS | 0.086 | 19 | 5.30448514283604e-09 | Yes |
| smooth_quantile | LSQR | 0.088 | 19 | 4.67630331701576e-09 | Yes |
| expectile | Spectral-MM (PCG) | 0.006 | 25 | 9.93240205970758e-09 | Yes |
| expectile | Spectral-MM (MM) | 0.010 | 33 | 9.58461678251744e-09 | Yes |
| expectile | CRLS | 0.006 | 5 | 1.32042275802581e-10 | Yes |
| expectile | CG | 0.006 | 5 | 1.1120564944783e-10 | Yes |
| expectile | LSMR | 0.006 | 5 | 1.32042279984667e-10 | Yes |
| expectile | CGLS | 0.006 | 5 | 1.11205740856982e-10 | Yes |
| expectile | LSQR | 0.006 | 5 | 1.11205746200004e-10 | Yes |
| pseudo_huber | Spectral-MM (PCG) | 0.008 | 26 | 6.1652900206107e-09 | Yes |
| pseudo_huber | Spectral-MM (MM) | 0.013 | 40 | 8.76329606205406e-09 | Yes |
| pseudo_huber | CRLS | 0.013 | 10 | 4.06815885620849e-09 | Yes |
| pseudo_huber | CG | 0.014 | 10 | 4.04272892956028e-09 | Yes |
| pseudo_huber | LSMR | 0.014 | 10 | 4.06815895463761e-09 | Yes |
| pseudo_huber | CGLS | 0.014 | 10 | 4.04272898754456e-09 | Yes |
| pseudo_huber | LSQR | 0.014 | 10 | 4.04272898085436e-09 | Yes |
| student_t | Spectral-MM (PCG) | 0.010 | 42 | 8.82286675717794e-09 | Yes |
| student_t | Spectral-MM (MM) | 0.014 | 35 | 6.26651895656438e-09 | Yes |
| student_t | CRLS | 0.024 | 25 | 6.22869255451275e-09 | Yes |
| student_t | CG | 0.025 | 25 | 7.01926967031514e-09 | Yes |
| student_t | LSMR | 0.025 | 25 | 6.22869257243692e-09 | Yes |
| student_t | CGLS | 0.026 | 25 | 7.01926970505876e-09 | Yes |
| student_t | LSQR | 0.026 | 25 | 7.01926967188238e-09 | Yes |

* indicates a warning or error; inspect the raw CSV. Actual BLAS parallelism depends on the linked provider.
