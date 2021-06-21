FUNCTION AMP_FWDFIT_STRUCTURE2ARRAY, srcstr, key


if key EQ 'multi' then begin
srcparm = FLTARR(6)

srcparm[0]   = srcstr[0].srcfwhm_max
srcparm[1]   = srcstr[0].srcflux
srcparm[2]   = srcstr[0].ecce
srcparm[3]   = srcstr[0].srcflux1
srcparm[4]   = srcstr[0].dist
srcparm[5]   = srcstr[0].pa

endif

RETURN, srcparm
END
