-- futhark run insp-data.fut < data/sahara.in > res.txt

-- entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
--                   (hfrac: f32) (lam: f32)
--                   (mappingindices : [N]i32)
--                   (images : [m][N]f32) =

entry main (X:[][]f32) (Xsqr:[][][]f32) (Xinv:[][][]f32) (beta0:[][]f32)
           (beta:[][]f32) (y_preds:[][]f32) (Nss:[]i32) (y_errors:[][]f32)
           (val_indss:[][]i32) (hs:[]i32) (nss:[]i32) (sigmas:[]f32) (MO_fsts:[]f32)
           (MOpp:[][]f32)  (MOp:[][]f32) (breaks:[]i32) (means:[]f32)

           (Xseq:[][]f32) (Xsqrseq:[][][]f32) (Xinvseq:[][][]f32)
           (beta0seq:[][]f32) (betaseq:[][]f32) (y_predsseq:[][]f32)
           (Nssseq:[]i32) (y_errorsseq:[][]f32) (val_indssseq:[][]i32)
           (hsseq:[]i32) (nssseq:[]i32) (sigmasseq:[]f32) (MO_fstsseq:[]f32)
           (MOppseq:[][]f32) (MOpseq:[][]f32) (breaksseq:[]i32) (meansseq:[]f32)
            =
-- , Xsqr, Xinv, beta0,
--                   beta, y_preds, Nss, y_errors,
--                   val_indss, hs, nss, sigmas,
--                   MO_fsts, MOs, MOs_NN, breaks, means

-- in (X, Xsqr, Xinv, beta0, beta, y_preds, Nss, y_errors, val_indss, hs, nss,
--     sigmas, MO_fsts, MOs, MOs_NN, breaks, means)

-- Kernel 1: X
let Xtfs = map2 (\x y ->
                 map2 (\x' y' -> f32.abs (x' - y') < 0.01) x y
                 ) X Xseq
let XallTrue = map (\x -> and x) Xtfs
               |> and

-- Kernel 2: Xsqr
let XsqrTfs = map2 (\x y ->
                        map2 (\x' y' ->
                              map2 (\x'' y'' -> f32.abs (x'' - y'') < 0.1) x' y'
                              ) x y
                        ) Xsqr  Xsqrseq

let XsqrAllTrue = map (\x -> map (\x' -> and x') x ) XsqrTfs
                  |> map (\x -> and x)
                  |> and

-- Kernel 3: Xinv
let XinvTfs = map2 (\x y ->
                        map2 (\x' y' ->
                              map2 (\x'' y'' -> f32.abs (x'' - y'') < 0.0000001) x' y'
                              ) x y
                        ) Xinv  Xinvseq

let XinvAllTrue = map (\x -> map (\x' -> and x') x ) XinvTfs
                  |> map (\x -> and x)
                  |> and

-- Kernel 4: beta0
let beta0tfs = map2 (\x y ->
                 map2 (\x' y' -> f32.abs (x' - y') < 1.1) x y
                 ) beta0 beta0seq
let beta0allTrue = map (\x -> and x) beta0tfs
               |> and

-- Kernel 5: beta
let betatfs = map2 (\x y ->
                 map2 (\x' y' -> f32.abs (x' - y') < 0.1) x y
                 ) beta betaseq
let betaallTrue = map (\x -> and x) betatfs
               |> and

-- Kernel 6: y_preds
let y_predstfs = map2 (\x y ->
                 map2 (\x' y' -> f32.abs (x' - y') < 0.1) x y
                 ) y_preds y_predsseq
let y_predsallTrue = map (\x -> and x) y_predstfs
               |> and

-- Kernel 7: Nss
let NssallTrue = map2 (\x' y' -> i32.abs (x' - y') < 1) Nss Nssseq |> and
--
let y_errorstfs = map2 (\x y ->
                        map2 (\x' y' -> if f32.isnan x'
                                        then true
                                        else f32.abs (x' - y') < 0.1) x y
                        ) y_errors y_errorsseq
let y_errorsallTrue = map (\x -> and x) y_errorstfs
                      |> and
--
let val_indsstfs = map2 (\x y ->
                        map2 (\x' y' -> x' == y') x y
                        ) val_indss val_indssseq
let val_indssallTrue = map (\x -> and x) val_indsstfs
                      |> and
-- Kernel 8
let hsallTrue = map2 (\x' y' -> x' == y') hs hsseq |> and
let nssallTrue = map2 (\x' y' -> x' == y') nss nssseq |> and
let sigmasallTrue = map2 (\x' y' -> f32.abs (x' - y') < 1) sigmas sigmasseq |> and
--  Kernel 9
let MO_fstsallTrue = map2 (\x' y' -> f32.abs (x' - y') < 1) MO_fsts MO_fstsseq |> and
-- Kernel 10
let MOpptfs = map2 (\x y ->
                        map2 (\x' y' -> if f32.isnan x'
                                        then true
                                        else f32.abs (x' - y') < 0.1) x y
                        ) MOpp MOppseq
let MOppallTrue = map (\x -> and x) MOpptfs
                      |> and
--
let MOptfs = map2 (\x y ->
                        map2 (\x' y' -> if f32.isnan x'
                                        then true
                                        else f32.abs (x' - y') < 0.1) x y
                        ) MOp MOpseq
let MOpallTrue = map (\x -> and x) MOptfs
                      |> and
--
let breaksallTrue = map2 (\x' y' -> x' == y') breaks breaksseq |> and
let meansallTrue = map2 (\x' y' -> f32.abs (x' - y') < 1) means meansseq |> and


in (XallTrue
   ,XsqrAllTrue
   ,XinvAllTrue
   ,beta0allTrue
   ,betaallTrue
   ,y_predsallTrue
   ,NssallTrue
   ,y_errorsallTrue
   ,val_indssallTrue
   ,hsallTrue
   ,nssallTrue
   ,sigmasallTrue
   ,MO_fstsallTrue
   ,MOppallTrue
   ,MOpallTrue
   ,breaksallTrue
   ,meansallTrue)
