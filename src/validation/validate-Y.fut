
-- Validating float arrays 

let epsilon = 0.1f32

let relError x y =
    let x' = f32.abs(x)
    let y' = f32.abs(y) in
    if f32.isnan x' && f32.isnan y' then true
    else
      if x' == 0.0f32 && y' == 0.0f32 then true
      else  let m = f32.max (f32.max x' y') 1.0
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


-- futhark run insp-data.fut < data/sahara.in > res.txt

entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) 
                  (imagesCU : [m][N]f32)
                  =

let valY   = validate1Dfloat images imagesCU

in valY









