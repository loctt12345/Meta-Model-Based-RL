1. step_per_epoch is too high, adjust to 10000
2. VAE epoch is too small, should be at least step_per_epoch/100 (10000 / 100)
3. why the decoder outputs r_mu, o_mu , r_logvar, o_logvar which are not used
