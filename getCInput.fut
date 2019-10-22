-- futhark opencl flatten.fut && ./flatten < data/sahara.in > data/flat.out
-- futhark opencl flatten.fut && ./flatten < data/peru.in > data/flat.out

entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =
                    
(trend, k, n, freq, hfrac, lam, m, N, mappingindices)
