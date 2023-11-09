;+
;
; NAME:
;   normalize_data
;
; PURPOSE:
;   This code normalize data in [0,1]^2
;
; CALLING SEQUENCE:
;   normalize_data, data, data1 = data1, second_set = second_set
;
; INPUTS:
;   data: the data to rescale [0,1]^2
;   data1: the reference set for which we have to rescale
;   second_set: usefull if more than one set needs to be rescaled
;
; OUTPUTS:
;   data_rescaled: the data rescaled in [0,1]^2
;
; HISTORY: Dec 2021, Perracchione E., created
;
function normalize_data, data, data1 = data1, second_set = second_set

  if KEYWORD_SET(data1) EQ 0 then begin
     data1 = data
     if KEYWORD_SET(second_set) then begin
       data_rescaled = data*max(second_set-min(second_set)) + min(second_set)
     endif else begin
       data_rescaled = (data-min(data1))/max(data1-min(data1))
     endelse
  endif else begin
  if KEYWORD_SET(second_set) then begin
    data_rescaled = data*max(second_set-min(second_set)) + min(second_set)
  endif else begin
    data_rescaled = (data-min(data1))/max(data1-min(data1))
  endelse
  endelse

  return, data_rescaled

end
