
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

let validate1Dint [n] (xs : [n]i32) (ys: [n]i32) : (bool, i32, i32, f32, f32) =
  let diffs = map3 (\x y i -> if (errori32 x y)
                              then (true, 0, 0, 0)
                              else (false, i, x, y)
                   ) xs ys (iota n)
  let num_inv = map (\(f,_,_,_) -> if f then 0 else 1) diffs
                  |> reduce (+) 0i32
  let (valid, ind, v1, v2) =
    reduce (\(b1, i1, x1, y1) (b2, i2, x2, y2) ->
                if b1 then (b2, i2, x2, y2)
                      else (b1, i1, x1, y1)
             )
             (true, 0, 0, 0) diffs
  in (valid, num_inv, ind, r32 v1, r32 v2)



let validate2Dint [n][m] (xs2 : [n][m]i32) (ys2: [n][m]i32) : (bool, i32, i32, f32, f32) =
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

entry main (X:[][]f32)          --0
           (Xsqr:[][][]f32)     --1
           (Xinv:[][][]f32)     --2
           (beta0:[][]f32)      --3
           (beta:[][]f32)       --4
           (y_preds:[][]f32)    --5
           (Nss:[]i32)          --6
           (y_errors:[][]f32)   --7
           (val_indss:[][]i32)  --8
           (nss:[]i32)          --9
           (hs:[]i32)           --10
           (sigmas:[]f32)       --11
           (MO_fsts:[]f32)
           (breaks:[]i32) 
           (means:[]f32)

           (Xseq:[][]f32)        --12
           (Xsqrseq:[][][]f32)   --13
           (Xinvseq:[][][]f32)   --14
           (beta0seq:[][]f32)    --15
           (betaseq:[][]f32)     --16
           (y_predsseq:[][]f32)  --17
           (Nssseq:[]i32)        --18
           (y_errorsseq:[][]f32) --19
           (val_indssseq:[][]i32)--20
           (nssseq:[]i32)        --21
           (hsseq:[]i32)         --22
           (sigmasseq:[]f32)     --23
           (MO_fstsseq:[]f32)    --24
           (breaksseq:[]i32) 
           (meansseq:[]f32)
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
let valNss   = validate1Dint Nss Nssseq
let valindss = validate2Dint val_indss val_indssseq
let valnss    = validate1Dint nss nssseq
let valhs     = validate1Dint hs hsseq
let valsigmas = validate1Dfloat sigmas sigmasseq
let valMOfst  = validate1Dfloat MO_fsts MO_fstsseq
let valbreaks = validate1Dint breaks breaksseq
let valmeans  = validate1Dfloat means meansseq

let fst xs = let (x,_,_,_,_) = xs in x
let snd xs = let (_,x,_,_,_) = xs in x
let thr xs = let (_,_,x,_,_) = xs in x
let frt xs = let (_,_,_,x,_) = xs in x
let fvt xs = let (_,_,_,_,x) = xs in x
let results = [valX, valXsqr, valXinv, valbeta0, valbeta, valyhat, valyerr,
               valNss, valindss, valnss, valhs, valsigmas, valMOfst, valbreaks, valmeans]

let allTestTrue  = map fst results
let allTestNumEr = map snd results
let allTestInds  = map thr results
let allTestFutE  = map frt results
let allTestOurE  = map fvt results

in (allTestTrue, allTestNumEr, allTestInds, allTestFutE, allTestOurE)


