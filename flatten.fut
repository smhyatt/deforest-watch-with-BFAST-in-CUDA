-- futhark opencl flatten.fut && ./flatten < data/sahara.in > data/flat.out
-- futhark opencl flatten.fut && ./flatten < data/peru.in > data/flat.out

entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
                  (hfrac: f32) (lam: f32)
                  (mappingindices : [N]i32)
                  (images : [m][N]f32) =

	let sample = map (\row -> map(\j -> let pixel = row[j] in
        if (f32.isnan pixel)
            then -340282346638528859811704183484516925440f32
            else row[j]
        ) (iota N)
    ) images[0:2]
    let flatbaby = flatten sample
in
(trend, k, n, freq, hfrac, lam, mappingindices, flatbaby)


