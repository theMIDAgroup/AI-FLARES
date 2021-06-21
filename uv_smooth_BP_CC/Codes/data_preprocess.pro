;+
;
; NAME:
;   data_preprocess
;
; PURPOSE:
;   This code creates the grid of points used to evaluate the interpolant
;
; CALLING SEQUENCE:
;   data_preprocess, usampl, Ulimit, N
;
; INPUTS:
;   usampl: the evaluation data (i.e. the vector defining the grid)
;   Ulimit: the range of the square cointaining the data in the uv-plane
;   N: number of grid data in one direction
;
; OUTPUTS:
;   epoints: the evaluation data
;

function data_preprocess, usampl, Ulimit, N

yy = replicate1(usampl, N)
xx = transpose(yy)

epointx = reform(xx,(N)^2,1)
epointy = reform(yy,(N)^2,1)

epointxx = (epointx+Ulimit)/(2*Ulimit)
epointyy = (epointy+Ulimit)/(2*Ulimit)

epoints = [[epointxx], [epointyy]]

return, epoints

end