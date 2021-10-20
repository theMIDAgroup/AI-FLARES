FUNCTION image_gradient, im, BACKWARD=backward
;+
 ; NAME: image_gradient
 ;
 ; PURPOSE: Gradient of an image using finite forward or backward differences
 ;
 ; EXPLANATION: Compute the gradient of an image using the finite forward differences (or finite backward differences 
 ; if the keyword BACKWARD is setted)
 ;
 ; CALLING SEQUENCE:
 ;
 ;       grad_fwd = image_grad(im)
 ;  or:
 ;       grad_bwd = image_grad(im, /BACKWARD)
 ; INPUTS:
 ; a = 2-D array (matrix) to which the gradient with the finite forward (or backward) differences formula is computed
 ; 
 ; OPTIONAL INPUT KEYWORDS:  
 ; /BACKWARD if the finite backward differences formula has to be computed
 ;
 ; Written: Dec 2019, Sabrina Guastavino (guastavino@dima.unige.it)
 ;-
  ; finite backward differences
  IF ( KEYWORD_SET( BACKWARD ) ) THEN BEGIN
  im_trasl_col_backward = im*0.
  im_trasl_col_backward[1:(size(im))[1]-1,*] = im[0:(size(im))[2]-2,*]
  im_backward_column = im-im_trasl_col_backward

  im_trasl_row_backward = im*0.
  im_trasl_row_backward[*,1:(size(im))[2]-1] = im[*,0:(size(im))[2]-2]
  im_backward_row = im-im_trasl_row_backward
  
  return, sqrt((im_backward_row)^2+(im_backward_column)^2)
  ENDIF ELSE BEGIN
    ;finite forwad differences
    im_trasl_col_forward = im*0.
    im_trasl_col_forward[0:(size(im))[1]-2,*] = im[1:(size(im))[2]-1,*]
    im_forward_column = im_trasl_col_forward-im

    im_trasl_row_forward = im*0.
    im_trasl_row_forward[*,0:(size(im))[2]-2] = im[*,1:(size(im))[2]-1]
    im_forward_row = im_trasl_row_forward-im

    return, sqrt((im_forward_row)^2+(im_forward_column)^2)
  ENDELSE
  
  end
