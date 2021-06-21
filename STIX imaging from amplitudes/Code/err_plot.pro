;
; $Id: errplot.pro,v 1.2 90/04/04 18:04:16 wave Exp $
;
Pro Err_plot, X, Low, High, Width = width, XDir = xDir, HMS= hms,  $
	COLOR = color, THICK = thick
;+
; NAME:
;	ERR_PLOT
; PURPOSE:
;	Overplot error bars over a previously drawn plot.
; CATEGORY:
;	J6 - plotting, graphics, one dimensional.
; CALLING SEQUENCE:
;	ERR_PLOT, Low, High	;X axis = point number
;	ERR_PLOT, X, Low, High	;to specify abscissae
; INPUTS:
;	Low = vector of lower estimates, = to data - error.
;	High = upper estimate, = to data + error.
; OPTIONAL INPUT PARAMETERS:
;	X = vector containing abscissae.
;	If the keyword Xdir is present, then this parameter is the y abscissae
; KEYWORD Parameters:
;	Width = width of bars, default = 1% of plot width.
;	Xdir = If present, the error bars are drawn horizontally.
;	HMS = if set, the (x-) axis is in date/time format.
;	COLOR = to specify another color than white
;	THICK = to specify line thickness
; OUTPUTS:
;	None.
; COMMON BLOCKS:
;	None.
; SIDE EFFECTS:
;	Overplot is produced.
; RESTRICTIONS:
;	Logarithmic restriction removed.
; PROCEDURE:
;	Error bars are drawn for each element.
;	For example:  Y = data values, ERR = symmetrical error estimates:
;		PLOT,Y	;Plot data
;		ERR_PLOT, Y-ERR, Y+ERR	;Overplot error bars.
;	If error estimates are non-symetrical:
;		PLOT,Y
;		ERR_PLOT, Upper, Lower	;Where upper & lower are bounds.
;	To plot versus a vector of abscissae:
;		PLOT,X,Y		;Plot data.
;		ERR_PLOT,X,Y-ERR,Y+ERR	;Overplot error estimates.
;
; MODIFICATION HISTORY:
;	DMS, RSI, June, 1983.
;	Joe Zawodney, LASP, Univ of Colo., March, 1986. Removed logarithmic
;		restriction.
;	DMS, March, 1989.  Modified for Unix WAVE.
;	June 1991, Modified for x error bars by A.Csillaghy
;	HMS keyword in Jul 93, A.Cs
;	COLOR Keyword in Sept 93, A.Csillaghy
;-

  on_error,2                      ;Return to caller if an error occurs
  if n_params(0) eq 3 then begin	;X specified?
    up = high
    down = low
    xx = x
  endif else begin	;Only 2 params
    up = x
    down = low
    xx=findgen(n_elements(up)) ;make our own x
  endelse

;  IF Keyword_Set( HMS ) THEN BEGIN
;    hms = 1 
;    xx =  x.julian
;    xRange = !pdt.dt_crange
;  ENDIF ELSE  BEGIN
    xRange = !x.crange
    hms = 0
;  ENDELSE

  if n_elements(width) eq 0 then width = .01 ;Default width
  width = width/2		;Centered
;
  n = n_elements(up) < n_elements(down) < n_elements(xx) ;# of pnts

  IF NOT Keyword_Set(XDIR) THEN BEGIN
    xxmin = min(xRange)	;X range
    xxmax = max(xRange)
    yymax = max(!y.crange)  ;Y range
    yymin = min(!y.crange)
  ENDIF ELSE BEGIN
    xxmin = min(!y.crange)	;X range
    xxmax = max(!y.crange)
    yymax = max(xRange)  ;Y range
    yymin = min(xRange)
  ENDELSE

  if !x.type eq 0 then begin	;Test for x linear
 ;Linear in x
    wid =  (xxmax - xxmin) * width ;bars = .01 of plot wide.
  endif else begin		;Logarithmic X
    xxmax = 10.^xxmax
    xxmin = 10.^xxmin
    wid  = (xxmax/xxmin)* width  ;bars = .01 of plot wide
  endelse
;
  IF NOT Keyword_Set( COLOR ) THEN color = !p.color
  IF NOT Keyword_Set( THICK ) THEN thick = !p.thick

  for i=0,n-1 do begin	;do each point.
    xxx = xx(i)	;x value
    if (xxx ge xxmin) and (xxx le xxmax) then begin
      IF down(i) NE up(i) THEN $
        IF Keyword_Set(XDir) THEN $
          IF hms THEN $
              oplot,Sec_To_DT( [down(i),down(i),down(i),up(i),up(i),up(i)] ),$
 	  [xxx-wid,xxx+wid,xxx,xxx,xxx-wid,xxx+wid], $
	COLOR = color , THICK = thick $
          ELSE $
              oplot, [down(i),down(i),down(i),up(i),up(i),up(i)] ,$
 	  [xxx-wid,xxx+wid,xxx,xxx,xxx-wid,xxx+wid], COLOR = color, $
	THICK = thick $
        ELSE  IF hms THEN $	  
          oplot, Jul_To_DT( [xxx-wid,xxx+wid,xxx,xxx,xxx-wid,xxx+wid] ),$
 	  [down(i),down(i),down(i),up(i),up(i),up(i)], COLOR = color, $
	THICK = thick $
        ELSE $
          oplot,[xxx-wid,xxx+wid,xxx,xxx,xxx-wid,xxx+wid],$
 	  [down(i),down(i),down(i),up(i),up(i),up(i)], COLOR = color, $
	THICK = thick
    endif
  endfor

  return
end
