
-- Validating float arrays

let epsilon = 0.01f32

let relError x y =
    let x' = f32.abs(x)
    let y' = f32.abs(y) in
    if f32.isnan x' && f32.isnan y' then true
    else
      if x' == 0.0f32 && y' == 0.0f32 then true
      else  let m = f32.max (f32.max x' y') 0.01
            in  f32.abs(x'-y') / m < epsilon

let validate1Dfloat [n] (xs : [n]f32) (ys: [n]f32) : (bool, i32, i32, f32, f32) =
  let diffs = map3 (\x y i -> if (relError x y)
                              then (true, 0, 0.0, 0.0)
                              else (false, i, x, y)
                   ) xs ys (iota n)
  let num_inv = map (\(f,_,_,_) -> if f then 0 else 1) diffs
                  |> reduce (+) 0i32
  let (valid, ind, v1, v2) =
      reduce (\(b1, i1, x1, y1) (b2, i2, x2, y2) ->
                if b1 then (b2, i2, x2, y2)
                      else (b1, i1, x1, y1)
             )
             (true, 0, 0.0, 0.0) diffs
  in  (valid, num_inv, ind, v1, v2)

let validate2Dfloat [n][m] (xs2 : [n][m]f32) (ys2: [n][m]f32) =
  let xs = flatten xs2
  let ys = flatten ys2
  in  validate1Dfloat xs ys


let validate3Dfloat [n][m][l] (xs3 : [n][m][l]f32) (ys3: [n][m][l]f32) =
  let xs2 = flatten xs3
  let ys2 = flatten ys3
  in  validate2Dfloat xs2 ys2


-- Validating int arrays

let errori32 x y =
    i32.abs(x - y) == 0

let validate1Dint [n] (xs : [n]i32) (ys: [n]i32) : (bool, i32, i32, i32) =
  let diffs = map3 (\x y i -> if (errori32 x y)
                              then (true, 0, 0, 0)
                              else (false, i, x, y)
                   ) xs ys (iota n)
  in  reduce (\(b1, i1, x1, y1) (b2, i2, x2, y2) ->
                if b1 then (b2, i2, x2, y2)
                      else (b1, i1, x1, y1)
             )
             (true, 0, 0, 0) diffs

let validate2Dint [n][m] (xs2 : [n][m]i32) (ys2: [n][m]i32) : (bool, i32, i32, i32) =
  let xs = flatten xs2
  let ys = flatten ys2
  in  validate1Dint xs ys


-- let validate3Dint [n][m][l] (xs3 : [n][m][l]i32) (ys3: [n][m][l]i32) : (bool, i32, i32, i32) =
--   let xs2 = flatten xs3
--   let ys2 = flatten ys3
--   in  validate2Dint xs2 ys2


-- let absError x y = f32.abs (x - y) < epsilon


-- entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
--                   (hfrac: f32) (lam: f32)
--                   (mappingindices : [N]i32)
--                   (images : [m][N]f32) =

entry main (X:[][]f32)
           (Xsqr:[][][]f32)
           (Xinv:[][][]f32)
           (beta0:[][]f32)
           (beta:[][]f32)
           (y_preds:[][]f32)
           (Nss:[]i32) (y_errors:[][]f32)(val_indss:[][]i32)
           --(hs:[]i32) (nss:[]i32) (sigmas:[]f32) (MO_fsts:[]f32)
        --    (MOpp:[][]f32)  (MOp:[][]f32) (breaks:[]i32) (means:[]f32)

           (Xseq:[][]f32)
           (Xsqrseq:[][][]f32)
           (Xinvseq:[][][]f32)
           (beta0seq:[][]f32)
           (betaseq:[][]f32)
           (y_predsseq:[][]f32)
           (Nssseq:[]i32) (y_errorsseq:[][]f32) (val_indssseq:[][]i32)
        --    (hsseq:[]i32) (nssseq:[]i32) (sigmasseq:[]f32) (MO_fstsseq:[]f32)
        --    (MOppseq:[][]f32) (MOpseq:[][]f32) (breaksseq:[]i32) (meansseq:[]f32)
            =
-- , Xsqr, Xinv, beta0,
--                   beta, y_preds, Nss, y_errors,
--                   val_indss, hs, nss, sigmas,
--                   MO_fsts, MOs, MOs_NN, breaks, means

-- in (X, Xsqr, Xinv, beta0, beta, y_preds, Nss, y_errors, val_indss, hs, nss,
--     sigmas, MO_fsts, MOs, MOs_NN, breaks, means)
let valX     = validate2Dfloat X Xseq
let valXsqr  = validate3Dfloat Xsqr Xsqrseq
let valXinv  = validate3Dfloat Xinv Xinvseq
let valbeta0 = validate2Dfloat beta0 beta0seq
let valbeta  = validate2Dfloat beta betaseq
let valyhat  = validate2Dfloat y_preds y_predsseq
let valyerr  = validate2Dfloat y_errors y_errorsseq
let valnss   = validate1Dint Nss Nssseq
let valindss = validate2Dint val_indss val_indssseq


let (vXs,nps,inds,fes,ves) = zip [valX,valXsqr]

in vXs


