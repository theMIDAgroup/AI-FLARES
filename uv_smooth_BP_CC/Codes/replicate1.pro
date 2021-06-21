;+
;
; NAME:
;   replicate1
;
; PURPOSE:
;   Replicate an array n times
;
; CALLING SEQUENCE:
;   replicate1, arr, n
;
; INPUTS:
;   arr:    an array
;   n:      number of times to replicate
;
; OUTPUTS:
;   arr1:  array consisting of arr replicated n times
;
;

function replicate1, arr, n

dim = size(arr, /dimensions)
if dim eq 0 then begin
  arr1 = replicate(arr, n)  
endif else begin

arr1 = fltarr(dim, n)

for i=0, n-1 do begin

arr1[*, i] = arr  
  
endfor
endelse
return, arr1

end