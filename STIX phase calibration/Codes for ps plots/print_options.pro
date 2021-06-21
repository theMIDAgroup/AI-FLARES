;+
;PROCEDURE: print_options
;PURPOSE:  controls postscript printing options
;KEYWORDS:
;  PORT:   print pages in portrait format (default)
;  LAND:   print pages in landscape format 
;  BW:     Use black and white mode  (untested)
;  COLOR:  Use Color postscript (default)
;FUTURE OPTIONS:
;  Ecapsulated postscript format
;  changing plotting area
;
;SEE ALSO:	"popen","pclose"
;
;CREATED BY:	Davin Larson
;LAST MODIFICATION:	@(#)print_options.pro	1.14 96/02/16
;-


pro print_options,      $
  PORTRAIT=port,   $
  LANDSCAPE=land,   $
  BW = bw,     $
  COLOR=col,   $
  ASPECT=aspect,  $
  XSIZE=xsize, $
  YSIZE=ysize, $
  FONT= font,  $
  PRINTER=printer, $
  DIRECTORY=printdir

@popen_com.pro
; Set defaults:
if n_elements(portrait) eq 0 then portrait=0
if n_elements(in_color) eq 0 then in_color=1
if n_elements(printer_name) eq 0 then printer_name=''
if n_elements(print_directory) eq 0 then print_directory=''
if n_elements(print_font) eq 0 then print_font = 0
if n_elements(print_aspect) eq 0 then print_aspect = 0

if keyword_set(land)    then  portrait= 0
if keyword_set(port)    then  portrait= 1
if keyword_set(col)     then  in_color= 1
if keyword_set(bw)      then  in_color= 0
if n_elements(printer)  ne 0 then  printer_name=printer
if n_elements(printdir) ne 0 then  print_directory=printdir
if n_elements(font)     ne 0 then  print_font = font
if n_elements(aspect)   ne 0 then  print_aspect=aspect

if !d.name eq 'PS' then begin
  aspect = print_aspect
  if keyword_set(aspect) then begin
    if portrait then scale=(8. < 10.5/aspect) else scale=(10.5 < 8./aspect)
    s = [1.0,aspect] * scale
    if portrait then offs =[(8.5-s(0))/2,11.-.5-s(1)] $
    else offs=[(8.5-s(1))/2,11.-(11.-s(0))/2]
    device,port=portrait,/inch,ysize=s(1),xsize=s(0),yoff=offs(1),xoff=offs(0) 
  endif else begin
    if portrait then begin
      if not keyword_set(xsize) then xsize = 7.0
      if not keyword_set(ysize) then ysize = 9.5
      xoff= (8.5 - xsize)/2
      yoff= (11. - ysize)/2
    endif else begin
      if not keyword_set(xsize) then xsize = 9.5
      if not keyword_set(ysize) then ysize = 7.0
      xoff= (8.5 - ysize)/2
      yoff= 11. - (11.-xsize)/2
    endelse
    device,port=portrait,/inches,ysize=ysize,yoff=yoff,xsize=xsize,xoff=xoff 
  endelse
  if in_color then device,/color,bits=8  $
  else   device,color=0
  !p.font = print_font
endif

return
end



