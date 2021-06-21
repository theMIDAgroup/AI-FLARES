function int_fun, x, a=a, b=b

  return, 1./(4.*a)*(2.*a*x*sqrt(4.*a^2.*x^2. + 1) + alog(2.*a*x + sqrt(4.*a^2.*x^2. + 1))) - b


end


PRO AMP_FWDFIT_SOURCE2MAP, srcstr, xyoffset, data, $
  pixel=pixel, mapsize=mapsize
 
  checkvar, pixel, [1., 1.]
  checkvar, mapsize, [128, 128]
  
  ; Define the map and its axes.
  data    = FLTARR(mapsize[0],mapsize[1])
  xy = Reform( ( Pixel_coord( [mapsize[0], mapsize[1]] ) ), 2, mapsize[0], mapsize[1] )
  x = reform(xy[0, *, *])*pixel[0] + xyoffset[0]
  y = reform(xy[1, *, *])*pixel[1] + xyoffset[1]
  
  
  
  nsrc    = N_ELEMENTS(srcstr)
  FOR n = 0, nsrc-1 DO BEGIN
    
    xcen = srcstr[n].srcx
    ycen = srcstr[n].srcy
    flux = srcstr[n].srcflux
    fwhm_max = srcstr[n].srcfwhm_max
    fwhm_min = srcstr[n].srcfwhm_min
    pa = srcstr[n].srcpa
    
    sinus=sin(pa*!dtor)
    cosinus=cos(pa*!dtor)


    dx =  x - xcen
    dy =  y - ycen

    x_tmp = (dx*cosinus) + (dy*sinus)
    y_tmp = -(dx*sinus) + (dy*cosinus)

    x_tmp = 2.*sqrt( 2.*alog(2.) )*x_tmp/fwhm_max
    y_tmp = 2.*sqrt( 2.*alog(2.) )*y_tmp/fwhm_min

    im_tmp = exp(-(x_tmp^2. + y_tmp^2.)/2.)

    data += im_tmp/(total(im_tmp)*pixel[0]*pixel[1])*flux
    
  ENDFOR
  
END


