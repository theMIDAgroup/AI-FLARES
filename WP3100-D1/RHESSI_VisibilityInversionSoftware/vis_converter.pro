FUNCTION vis_converter, energymin, energymax, u, v, realvis, imvis, err, time_interval, xyoffset

nv=n_elements(u)
vis         = REPLICATE( {hsi_vis}, nv )
vis.u       = u
vis.v       = v
vis.obsvis  = COMPLEX(realvis,imvis)
spatfreq    = SQRT(vis.u^2 + vis.v^2)
res         = 0.5 / spatfreq
vis.isc     = ROUND(ALOG(res/2.23)/ALOG(3)*2.)
vis.chi2    = 1
vis.sigamp  = err
vis.harm    = 1
vis.erange  = [energymin,energymax]
vis.trange  = time_interval
vis.xyoffset= xyoffset

RETURN, vis

END