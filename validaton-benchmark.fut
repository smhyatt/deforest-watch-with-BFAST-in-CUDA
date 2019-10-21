-- futhark run insp-data.fut < data/sahara.in > res.txt

-- entry main [m][N] (trend: i32) (k: i32) (n: i32) (freq: f32)
--                   (hfrac: f32) (lam: f32)
--                   (mappingindices : [N]i32)
--                   (images : [m][N]f32) =

entry main X:[][]f32, Xseq:[][]f32 =
-- , Xsqr, Xinv, beta0,
--                   beta, y_preds, Nss, y_errors,
--                   val_indss, hs, nss, sigmas,
--                   MO_fsts, MOs, MOs_NN, breaks, means

-- in (X, Xsqr, Xinv, beta0, beta, y_preds, Nss, y_errors, val_indss, hs, nss,
--     sigmas, MO_fsts, MOs, MOs_NN, breaks, means)

(X,Xseq)
