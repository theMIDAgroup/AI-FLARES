PRO amp_src_structure__define
  ;
  ; Defines a structure that defines one component of a source structure used by vis_fwdfit and related modules.
  ;
  ; {vis_source_structure} returns the following structure
  ;
  ; 9-Dec-2005  Initial version (ghurford@ssl.berkeley.edu)
  ; 30-Oct-13 A.M.Massone   Removed hsi dependencies
  ;
  dummy = {amp_src_structure,   $
    srctype:      ' ',      $   ; Label indicating source type
    srcflux:     0.,      $   ; Semi-calibrated flux (ph/cm2/s)
    srcx:        0.,      $   ; X-offset (+W) relative to sun center (arcsec)
    srcy:        0.,      $   ; Y-offset (+N) relative to sun center (arcsec)
    srcfwhm_max:     0.,      $   ; Source FWHM diameter (arcsec)
    srcfwhm_min:     0.,      $   ; Source FWHM diameter (arcsec)
    srcpa:       0.,      $   ; Position angle of long axis (degrees E of N)
    loop_angle:    0.}  ; Angle subtended by loop, as seen from its center of curvature (deg)
  RETURN
END
