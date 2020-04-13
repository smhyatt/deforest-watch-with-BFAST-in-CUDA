-- BFAST-irregular: version handling obscured observations (e.g., clouds)
-- ==
-- compiled input @ data/sahara.in.gz
-- output @ data/sahara.out.gz

-- compiled input @ data/testin.in

-- output @ data/test.out

-- compiled input @ data/D1.in.gz
-- compiled input @ data/D2.in.gz
-- compiled input @ data/peru.in.gz

-- output @ data/peru.out.gz



-- if (x>2.71) then log_10 x else 1
let logplus (x: f32) : f32 =
  if x > (f32.exp 1)
  then f32.log x else 1

let adjustValInds [N] (n : i32) (ns : i32) (Ns : i32) (val_inds : [N]i32) (ind: i32) : i32 =
    if ind < Ns - ns then (unsafe val_inds[ind+ns]) - n else -1
-- returns some sort of indexing (for a scatter).


-- filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
-- Input:   p:(p->value:true or nan:false) dummy:nan arr:[nan,float,nan,float]
-- Returns: ([(float,int),(float,int)],int)
let filterPadWithKeys [n] 't
           (p : (t -> bool))
           (dummy : t)
           (arr : [n]t) : ([n](t,i32), i32) =
  -- [0,1,0,1] <- [nan,float,nan,float]
  let tfs = map (\a -> if p a then 1 else 0) arr
  -- number of valid
  let isT = scan (+) 0 tfs
  let i   = last isT
  -- isT:  [0,1,1,2]
  -- inds: [-1,0,-1,1]
  let inds= map2 (\a iT -> if p a then iT-1 else -1) arr isT
  --X [nan,nan,nan,nan]
  --I inds: [-1,0,-1,1]
  --D [nan,float,nan,float]
  --R [float,float,nan,nan]
  let rs  = scatter (replicate n dummy) inds arr
  --X [0,0,0,0]
  --I inds: [-1,0,-1,1]
  --D [0,1,2,3]
  --R [1,3,0,0]
  let ks  = scatter (replicate n 0) inds (iota n)
  in  (zip rs ks, i)

-- | builds the X matrices; first result dimensions of size 2*k+2
let mkX_with_trend [N] (k2p2: i32) (f: f32) (mappingindices: [N]i32): [k2p2][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                else if i == 1 then r32 ind
                else let (i', j') = (r32 (i / 2), r32 ind)
                     let angle = 2f32 * f32.pi * i' * j' / f
                     in  if i % 2 == 0 then f32.sin angle
                                       else f32.cos angle
            ) mappingindices
      ) (iota k2p2)

let mkX_no_trend [N] (k2p2m1: i32) (f: f32) (mappingindices: [N]i32): [k2p2m1][N]f32 =
  map (\ i ->
        map (\ind ->
                if i == 0 then 1f32
                          else let i = i + 1
		        let (i', j') = (r32 (i / 2), r32 ind)
                let angle = 2f32 * f32.pi * i' * j' / f
                in  if i % 2 == 0 then f32.sin angle
                                  else f32.cos angle
            ) mappingindices
      ) (iota k2p2m1)

---------------------------------------------------
-- Adapted matrix inversion so that it goes well --
-- with intra-blockparallelism                   --
---------------------------------------------------

  let gauss_jordan [nm] (n:i32) (m:i32) (A: *[nm]f32): [nm]f32 =
    loop A for i < n do
      let v1 = A[i]
      let A' = map (\ind -> let (k, j) = (ind / m, ind % m)
                            in if v1 == 0.0 then unsafe A[k*m+j] else
                            let x = unsafe (A[j] / v1) in
                                if k < n-1  -- Ap case
                                then unsafe ( A[(k+1)*m+j] - A[(k+1)*m+i] * x )
                                else x      -- irow case
                   ) (iota nm)
      in  scatter A (iota nm) A'

  let mat_inv [n] (A: [n][n]f32): [n][n]f32 =
    let m = 2*n -- 2*K = 16
    let nm= n*m -- 8*16 = 128
    -- Pad the matrix with the identity matrix.
    --                        (7,15)       =  127 / 16, 127 % 16
    --                        (7,15)       =  127 / 16, 127 % 16
    let Ap = map (\ind -> let (i, j) = (ind / m, ind % m)
                          in  if j < n then unsafe ( A[i,j] )
                                       else if j == n+i
                                            then 1.0
                                            else 0.0
                 ) (iota nm)
    let Ap' = gauss_jordan n m Ap
    -- Drop the identity matrix at the front!
    in (unflatten n m Ap')[0:n,n:2*n]
--------------------------------------------------
--------------------------------------------------

let dotprod [n] (xs: [n]f32) (ys: [n]f32): f32 =
  reduce (+) 0.0 <| map2 (*) xs ys

let matvecmul_row [n][m] (xss: [n][m]f32) (ys: [m]f32) =
  map (dotprod ys) xss

let dotprod_filt [n] (vct: [n]f32) (xs: [n]f32) (ys: [n]f32) : f32 =
  f32.sum (map3 (\v x y -> x * y * if (f32.isnan v) then 0.0 else 1.0) vct xs ys)

let matvecmul_row_filt [n][m] (xss: [n][m]f32) (ys: [m]f32) =
    map (\xs -> map2 (\x y -> if (f32.isnan y) then 0 else x*y) xs ys |> f32.sum) xss

let matmul_filt [n][p][m] (xss: [n][p]f32) (yss: [p][m]f32) (vct: [p]f32) : [n][m]f32 =
  map (\xs -> map (dotprod_filt vct xs) (transpose yss)) xss

----------------------------------------------------
----------------------------------------------------

-- | implementation is in this entry point
--   the outer map is distributed directly
entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
  ----------------------------------
  -- 1. make interpolation matrix --
  ----------------------------------
  let k2p2 = 2*k + 2
  let k2p2' = if trend > 0 then k2p2 else k2p2-1
  let X = intrinsics.opaque <|
	  if trend > 0
          then mkX_with_trend k2p2' freq mappingindices
	  else mkX_no_trend   k2p2' freq mappingindices


  -- PERFORMANCE BUG: instead of `let Xt = copy (transpose X)`
  --   we need to write the following ugly thing to force manifestation:
  let zero = r32 <| (N*N + 2*N + 1) / (N + 1) - N - 1
  let Xt  = intrinsics.opaque <|
            map (map (+zero)) (copy (transpose X))

  let Xh  =  (X[:,:n])
  let Xth =  (Xt[:n,:])
  let Yh  =  (images[:,:n])

  ----------------------------------
  -- 2. mat-mat multiplication    --
  ----------------------------------
  let Xsqr = intrinsics.opaque <|
             map (matmul_filt Xh Xth) Yh

  ----------------------------------
  -- 3. matrix inversion          --
  ----------------------------------
  let Xinv = intrinsics.opaque <|
             map mat_inv Xsqr
  ---------------------------------------------
  -- 4. several matrix-vector multiplication --
  ---------------------------------------------
  let beta0  = map (matvecmul_row_filt Xh) Yh   -- [2k+2]
               |> intrinsics.opaque

  let beta   = map2 matvecmul_row Xinv beta0    -- [2k+2]
               |> intrinsics.opaque -- ^ requires transposition of Xinv
                                    --   unless all parallelism is exploited

  let y_preds= map (matvecmul_row Xt) beta      -- [N]
               |> intrinsics.opaque -- ^ requires transposition of Xt (small)
                                    --   can be eliminated by passing
                                    --   (transpose X) instead of Xt

  ---------------------------------------------
  -- 5. filter etc.                          --
  ---------------------------------------------
  let (Nss, y_errors, val_indss) = ( intrinsics.opaque <| unzip3 <|
    -- y p
    map2 (\y y_pred ->
            let y_error_all = zip y y_pred |>
                map (\(ye,yep) -> if !(f32.isnan ye)
                                  then ye-yep else f32.nan )
            -- [nan,dif,nan,dif]
            let (tups, Ns) = filterPadWithKeys (\y -> !(f32.isnan y)) (f32.nan) y_error_all
            -- (tups:([false,],[nan,dif,nan,dif]), Ns:float#ofvalid)
            let (y_error, val_inds) = unzip tups
            in  (Ns, y_error, val_inds)
         ) images y_preds )

    -- outout: (2,[float,float,nan,nan],[1,3,0,0])
  ------------------------------------------------
  -- 6. ns and sigma (can be fused with above)  --
  ------------------------------------------------
  let (hs, nss, sigmas) = intrinsics.opaque <| unzip3 <|
    map2 (\yh y_error ->
            let ns    = map (\ye -> if !(f32.isnan ye) then 1 else 0) yh
                        |> reduce (+) 0
            let sigma = map (\i -> if i < ns then unsafe y_error[i] else 0.0) (iota n)
                        |> map (\ a -> a*a ) |> reduce (+) 0.0
            let sigma = f32.sqrt ( sigma / (r32 (ns-k2p2)) )
            let h     = t32 ( (r32 ns) * hfrac )
            in  (h, ns, sigma)
         ) Yh y_errors

  -- map2 (\sample diff ->
  --                    ns = map (\pixel -> [0,1,0,1]) -> reduce -> 2
  --                    sigma = [float,float,0.0,0.0] -> [float^2,float^2,0.0,0.0] -> float
  --                    sigma = float
  --                    h = int
  -- output: ([int,int,int,int],[valid0,valid1,valid2,valid3]:float,[float,float,flaot])
  ---------------------------------------------
  -- 7. moving sums first and bounds:        --
  ---------------------------------------------
  -- ker9
  let hmax = reduce_comm (i32.max) 0 hs
  -- let hmax_pad = n * hfrac
  -- find max h value
  let MO_fsts = zip3 y_errors nss hs |>
    map (\(y_error, ns, h) -> unsafe
            map (\i -> if i < h then unsafe y_error[i + ns-h+1] else 0.0) (iota hmax) --(iota hmax_pad)
            -- [float,float,0,0]
            |> reduce (+) 0.0
            -- float
        )
        |> intrinsics.opaque
  -- [float,float,...,m]

  -- ker10
  let BOUND = map (\q -> let t   = n+1+q
                         let time = unsafe mappingindices[t-1]
                         let tmp = logplus ((r32 time) / (r32 mappingindices[N-1]))
                         in  lam * (f32.sqrt tmp)
                  ) (iota (N-n))
  -- [0,1,2,3,...,n]
  -- [114+0,114+1,...,114+n]
  -- time = [int,int,...]
  -- tmp:float
  -- lam:float that is our bound.

  ---------------------------------------------
  -- 8. moving sums computation:             --
  ---------------------------------------------
  let (MOs, MOs_NN, breaks, means) = zip (zip4 Nss nss sigmas hs) (zip3 MO_fsts y_errors val_indss) |>
    -- (Nss:[#valid,#valid,...,N], nss:[#valid,#valid,...,n], sigmas:[float,float,...,n], hs[int,int,...,n])
    -- (Ns:valid-float, ns:valid-float, sigma:float, h:int)
    -- (MO_fsts:[float,float,...,hmax], y_errors:[float,float,...,nan,nan,...,N], val_indss:[1,3,..,0,0,..,N])
    -- (MO_fst:float, y_error:float/nan, val_inds:int)
    map (\ ( (Ns,ns,sigma, h), (MO_fst,y_error,val_inds) ) ->
            let Nmn = N-n   -- Nmn:last part of the timeline
            let MO = map (\j -> if j >= Ns-ns then 0.0
                                else if j == 0 then MO_fst
                                else unsafe (-y_error[ns-h+j] + y_error[ns+j])
                         ) (iota Nmn) |> scan (+) 0.0

            -- [0,1,2,3,...,Nmn]
            -- Makes some signal processing of the timeline/errors/predictions.
            -- MO:[float,float,..,Nmn] (an accumulated value based on y_errors, Ns, ns, MO_fst and h.)

            let MO' = map (\mo -> mo / (sigma * (f32.sqrt (r32 ns))) ) MO
            -- MO':[float,float,..,Nmn] (a new list of MO's that have been futher computed applying sigma and h.)

	        let (is_break, fst_break) =
		    map3 (\mo' b j ->  if j < Ns - ns && !(f32.isnan mo')
				      then ( (f32.abs mo') > b, j )
				      else ( false, j )
		         ) MO' BOUND (iota Nmn)
		        |> reduce (\ (b1,i1) (b2,i2) ->
                                if b1 then (b1,i1)
                                else if b2 then (b2, i2)
                                else (b1,i1)
              	      	     ) (false, -1)

          -- MO':[float,float,..,Nmn] BOUND:[float,float,..,(N-n)] [j=0,1,2,3,..,Nmn]
          -- map3:[(true,0),(false,1),(true,2),(false,3),...,(true,LEN)]
          -- reduce:[(true,0),(true,0),(true,2),(true,2),...,(true,LEN)] -> (is_break:true,fst_break:LEN) -> LEN???

	        let mean = map2 (\x j -> if j < Ns - ns then x else 0.0 ) MO' (iota Nmn)
			    |> reduce (+) 0.0

          -- MO':[float,float,..,Nmn] [j=0,1,2,3,..,Nmn]
          -- mean: map2:   [x,x,x,...,0.0,0.0,Nmn]
          -- mean: reduce: float

	        let fst_break' = if !is_break then -1
                             else let adj_break = adjustValInds n ns Ns val_inds fst_break
                                  in  ((adj_break-1) / 2) * 2 + 1  -- Cosmin's validation hack
            let fst_break' = if ns <=5 || Ns-ns <= 5 then -2 else fst_break'

            -- let val_inds' = map (adjustValInds n ns Ns val_inds) (iota Nmn)
            -- let MO'' = scatter (replicate Nmn f32.nan) val_inds' MO'
            in (MO', MO', fst_break', mean)

            -- 1:fst_break':[int,-1,int,-1,...,LEN]
            -- 2:fst_break':[int,-2,int,...,-2,-1,...,LEN]
            -- val_inds':[int,int,int,...,Nmn] (another indexing)
            -- MO':[float,float,..,Nmn]
            -- MO'':[MO',MO',nan,nan,MO',...,Nmn]

        ) |> unzip4

        -- return:(MO'':[MO',MO',nan,nan,MO',...,Nmn], MO':[float,float,..,Nmn],
        --         fst_break':[int,-2,int,...,-2,-1,...,LEN], mean:[float,float,...,Nmn])


  ------------------------------------------------------------------------------
  -- This is the original out of this file
--   in (breaks, means)
  -- This is the final validation output
--   in (X, Xsqr, Xinv, beta0, beta, y_preds, Nss, y_errors, val_indss, hs, nss, sigmas, MO_fsts, MOs, MOs_NN, breaks, means)
  -- This is the working validation output
  in (X, Xsqr, Xinv, beta0, beta, y_preds, Nss, y_errors, val_indss, nss, hs, sigmas, MO_fsts, breaks, means)




  -- (breaks:fst_break':[int,-2,int,...,-2,-1,...,LEN], means:mean:[float,float,...,Nmn])



-- For Fabian: with debugging info, replace the result with the next line
--in (MO_fsts, Nss, nss, sigmas, _MOs, _MOs_NN, BOUND, breaks, means, y_errors, y_preds)

-- gcc -O2 --std=c99 bfast-cloudy-wip.c -lOpenCL -lm
-- FUTHARK_INCREMENTAL_FLATTENING=1 ~/WORK/gits/futhark/tools/futhark-autotune --compiler=futhark-opencl --pass-option=--default-tile-size=8 --stop-after 1500 --calc-timeout bfast-cloudy-wip.fut --compiler=futhark-opencl

