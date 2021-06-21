function amp_fwdfit, type, fun_name, ampobs, sigamp, u, v, n_free, $
                     SwarmSize = SwarmSize, TolFun = TolFun, maxiter = maxiter, uncertainty = uncertainty

default, SwarmSize, 100.
default, TolFun, 1e-06

ampobs = transpose(cmreplicate(ampobs, SwarmSize))
sigamp = transpose(cmreplicate(sigamp, SwarmSize))
extra = {type: type, $
         ampobs: ampobs, $
         sigamp: sigamp, $
         u: u, $
         v: v, $
         n_free: n_free}

estimate_flux = max(ampobs)

if type eq 'circle' then begin

  lb = [0.1*estimate_flux, 1.]
  ub = [1.5*estimate_flux, 100.]
  Nvars = n_elements(lb)

  if ~keyword_set(maxiter) then maxiter = Nvars*SwarmSize

  optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
  xopt = optim_f.xopt
  
  srcstr = {amp_src_structure}
  srcstr.srctype ='circle'

  fitsigmas = {amp_src_structure}
  fitsigmas.srctype ='std.dev'

  srcstr.srcflux         = xopt[0]
  srcstr.srcfwhm_max     = xopt[1]
  srcstr.srcfwhm_min     = xopt[1]

  if keyword_set(uncertainty) then begin
    
    print, ' '
    print, 'Uncertainty: '
    print, ' 
    
    ntry = 20
    namp = N_ELEMENTS(ampobs)

    trial_results = fltarr(2, ntry)
    for n=0,ntry-1 do begin
      testerror           = RANDOMN(iseed, namp)          ; nvis element vector normally distributed with sigma = 1
      amptest           = ampobs + testerror * sigamp
      
      extra = {type: type, $
        ampobs: amptest, $
        sigamp: sigamp, $
        u: u, $
        v: v, $
        n_free: n_free}
      
      optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
      xopt = optim_f.xopt

      trial_results[*,n]  = xopt

    endfor

    std_dev_par = stddev(trial_results, dimension=2)

    fitsigmas.srcflux         = std_dev_par[0]
    fitsigmas.srcfwhm_max     = std_dev_par[1]
    fitsigmas.srcfwhm_min     = std_dev_par[1]

  endif
  
endif


if type eq 'ellipse' then begin
  
  lb = [0.1*estimate_flux, 1., 0., -5.]
  ub = [1.5*estimate_flux, 100., 1., 5.]
  Nvars = n_elements(lb)

  if ~keyword_set(maxiter) then maxiter = Nvars*SwarmSize

  optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
  xopt = optim_f.xopt

  srcstr = {amp_src_structure}
  srcstr.srctype ='ellipse'

  fitsigmas = {amp_src_structure}
  fitsigmas.srctype ='std.dev'

  srcstr.srcflux = xopt[0]

  ecmsr = REFORM(SQRT(xopt[2]^2 + xopt[3]^2))
  eccen = SQRT(1 - EXP(-2*ecmsr))

  IF ecmsr GT 0 THEN srcstr.srcpa = reform(ATAN(xopt[3], xopt[2]) * !RADEG) + 90.

  srcstr.srcfwhm_min = xopt[1] * (1-eccen^2)^0.25
  srcstr.srcfwhm_max = xopt[1] / (1-eccen^2)^0.25
  

  if keyword_set(uncertainty) then begin

    print, ' '
    print, 'Uncertainty: '
    print, '

    ntry = 20
    namp = N_ELEMENTS(ampobs)
    
    trial_results = fltarr(Nvars, ntry)
    for n=0,ntry-1 do begin
      testerror           = RANDOMN(iseed, namp)          ; nvis element vector normally distributed with sigma = 1
      amptest           = ampobs + testerror * sigamp

      extra = {type: type, $
        ampobs: amptest, $
        sigamp: sigamp, $
        u: u, $
        v: v, $
        n_free: n_free}

      optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
      xopt = optim_f.xopt
      

      ecmsr = REFORM(SQRT(xopt[2]^2 + xopt[3]^2))
      eccen = SQRT(1 - EXP(-2*ecmsr))
        
        IF ecmsr GT 0 THEN trial_results[3,n] = reform(ATAN(xopt[3], xopt[2]) * !RADEG) + 90.
        
        trial_results[0,n]  = xopt[0]
        trial_results[1,n]  = xopt[1] / (1-eccen^2)^0.25
        trial_results[2,n]  = xopt[1] * (1-eccen^2)^0.25
        
      endfor

      fitsigmas.srcflux         = stddev(trial_results[0, *])
      fitsigmas.srcfwhm_max     = stddev(trial_results[1, *])
      fitsigmas.srcfwhm_min     = stddev(trial_results[2, *])
      avsrcpa                   = ATAN(TOTAL(SIN(trial_results[3, *] * !DTOR)), $
                                  TOTAL(COS(trial_results[3, *] * !DTOR))) * !RADEG
      groupedpa                   = (810 + avsrcpa - trial_results[3, *]) MOD 180. 
      fitsigmas.srcpa          = STDDEV(groupedpa)

  endif

endif


if type EQ 'multi' then begin
  
  lb = [0.,  0.1*estimate_flux, 1e-3,  0.1*estimate_flux, 0., 0.]
  ub = [100., 1.5*estimate_flux, 1., 1.5*estimate_flux, 30., 180.]
  Nvars = n_elements(lb)
  
  if ~keyword_set(maxiter) then maxiter = Nvars*SwarmSize
  
  Nruns = 20
  xx_opt = []
  f = fltarr(Nruns)
  
  for i = 0,Nruns-1 do begin
    optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
    f[i] = optim_f.fopt
    xx_opt = [[xx_opt],optim_f.xopt]
  endfor
  
    dummy = min(f,location)
    xopt = xx_opt(location,*)
    
    srcstr = {amp_src_structure}
    srcstr.srctype ='ellipse'
    srcstr = amp_fwdfit_bifurcate(srcstr)
    
    fitsigmas = {amp_src_structure}
    fitsigmas.srctype ='std.dev'
    fitsigmas = amp_fwdfit_bifurcate(fitsigmas)
    
    srcstr[0].srcflux     = xopt[1]
    srcstr[0].srcfwhm_max     = xopt[0]
    srcstr[0].srcfwhm_min     = xopt[0]
    srcstr[0].srcx     = xopt[4] * cos(xopt[5] * !dtor)
    srcstr[0].srcy     = xopt[4] * sin(xopt[5] * !dtor)

    srcstr[1].srcflux     = xopt[3]
    srcstr[1].srcfwhm_max     = xopt[0]*xopt[2]
    srcstr[1].srcfwhm_min     = xopt[0]*xopt[2]
    srcstr[1].srcx     = -xopt[4] * cos(xopt[5] * !dtor)
    srcstr[1].srcy     = -xopt[4] * sin(xopt[5] * !dtor)
    
    
    if keyword_set(uncertainty) then begin

    print, ' '
    print, 'Uncertainty: '
    print, '

    ntry = 20
    namp = N_ELEMENTS(ampobs)

    trial_results = fltarr(Nvars, ntry)
    for n=0,ntry-1 do begin
      nn = n
      testerror           = RANDOMN(nn, namp);RANDOMN(iseed, namp)          ; nvis element vector normally distributed with sigma = 1
      amptest           = ampobs + testerror * sigamp

      extra = {type: type, $
        ampobs: amptest, $
        sigamp: sigamp, $
        u: u, $
        v: v, $
        n_free: n_free}
        
      xx_opt = []
      f = fltarr(Nruns)

      for i = 0,Nruns-1 do begin
        optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
        f[i] = optim_f.fopt
        xx_opt = [[xx_opt],optim_f.xopt]
      endfor

      dummy = min(f,location)
      xopt = xx_opt(location,*)
        
      trial_results[0, n] = xopt[1]
      trial_results[1, n] = xopt[0]
      trial_results[2, n] = xopt[4] * cos(xopt[5] * !dtor)
      trial_results[3, n] = xopt[4] * sin(xopt[5] * !dtor)

      trial_results[4, n] = xopt[3]
      trial_results[5, n] = xopt[0]*xopt[2]
  
     endfor
     
     fitsigmas[0].srcflux     = stddev(trial_results[0, *])
     fitsigmas[0].srcfwhm_max     = stddev(trial_results[1,*])
     fitsigmas[0].srcfwhm_min     = stddev(trial_results[1,*])
     fitsigmas[0].srcx     = stddev(trial_results[2,*])
     fitsigmas[0].srcy     = stddev(trial_results[3,*])

     fitsigmas[1].srcflux     = stddev(trial_results[4,*])
     fitsigmas[1].srcfwhm_max     = stddev(trial_results[5,*])
     fitsigmas[1].srcfwhm_min     = stddev(trial_results[5,*])
     fitsigmas[1].srcx     = stddev(trial_results[2,*])
     fitsigmas[1].srcy     = stddev(trial_results[3,*])
        
     endif

endif

if type EQ 'multi_fl' then begin

  lb = [1.,  0.1*estimate_flux, 1e-3,  1., 0., 0.]
  ub = [100., 1.5*estimate_flux, 1., 100., 30., 180.]
  Nvars = n_elements(lb)

  if ~keyword_set(maxiter) then maxiter = Nvars*SwarmSize

  Nruns = 20
  xx_opt = []
  f = fltarr(Nruns)

  for i = 0,Nruns-1 do begin
    optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
    f[i] = optim_f.fopt
    xx_opt = [[xx_opt],optim_f.xopt]
  endfor

  dummy = min(f,location)
  xopt = xx_opt(location,*)

  srcstr = {amp_src_structure}
  srcstr.srctype ='ellipse'
  srcstr = amp_fwdfit_bifurcate(srcstr)

  fitsigmas = {amp_src_structure}
  fitsigmas.srctype ='std.dev'
  fitsigmas = amp_fwdfit_bifurcate(fitsigmas)

  srcstr[0].srcflux     = xopt[1]
  srcstr[0].srcfwhm_max     = xopt[0]
  srcstr[0].srcfwhm_min     = xopt[0]
  srcstr[0].srcx     = xopt[4] * cos(xopt[5] * !dtor)
  srcstr[0].srcy     = xopt[4] * sin(xopt[5] * !dtor)

  srcstr[1].srcflux     = xopt[1]*xopt[2]
  srcstr[1].srcfwhm_max     = xopt[3]
  srcstr[1].srcfwhm_min     = xopt[3]
  srcstr[1].srcx     = -xopt[4] * cos(xopt[5] * !dtor)
  srcstr[1].srcy     = -xopt[4] * sin(xopt[5] * !dtor)


  if keyword_set(uncertainty) then begin

    print, ' '
    print, 'Uncertainty: '
    print, '

    ntry = 20
    namp = N_ELEMENTS(ampobs)

    trial_results = fltarr(Nvars, ntry)
    for n=0,ntry-1 do begin
      testerror           = RANDOMN(iseed, namp)          ; nvis element vector normally distributed with sigma = 1
      amptest           = ampobs + testerror * sigamp

      extra = {type: type, $
        ampobs: amptest, $
        sigamp: sigamp, $
        u: u, $
        v: v, $
        n_free: n_free}

      xx_opt = []
      f = fltarr(Nruns)

      for i = 0,Nruns-1 do begin
        optim_f = swarmintelligence(fun_name, Nvars, lb, ub, SwarmSize, TolFun, maxiter, extra = extra)
        f[i] = optim_f.fopt
        xx_opt = [[xx_opt],optim_f.xopt]
      endfor

      dummy = min(f,location)
      xopt = xx_opt(location,*)

      trial_results[0, n] = xopt[1]
      trial_results[1, n] = xopt[0]
      trial_results[2, n] = xopt[4] * cos(xopt[5] * !dtor)
      trial_results[3, n] = xopt[4] * sin(xopt[5] * !dtor)

      trial_results[4, n] = xopt[1]*xopt[2]
      trial_results[5, n] = xopt[3]

    endfor

    fitsigmas[0].srcflux     = stddev(trial_results[0, *])
    fitsigmas[0].srcfwhm_max     = stddev(trial_results[1,*])
    fitsigmas[0].srcfwhm_min     = stddev(trial_results[1,*])
    fitsigmas[0].srcx     = stddev(trial_results[2,*])
    fitsigmas[0].srcy     = stddev(trial_results[3,*])

    fitsigmas[1].srcflux     = stddev(trial_results[4,*])
    fitsigmas[1].srcfwhm_max     = stddev(trial_results[5,*])
    fitsigmas[1].srcfwhm_min     = stddev(trial_results[5,*])
    fitsigmas[1].srcx     = stddev(trial_results[2,*])
    fitsigmas[1].srcy     = stddev(trial_results[3,*])

  endif

endif  

return, {srcstr: srcstr, fitsigmas: fitsigmas}

end