;pro demo_for_desaturation
;;; Demo for the desaturation procedure for AIA images

;;; Cutout data request line.

;ssw_cutout_service,'2017/09/10 15:45:00','2017/09/10 16:15:00',ref_helio='S09W91',fovx=350,fovy=350,email='guastavino@dima.unige.it',waves=[94,131,171,193],max_frames=1000,instrument='aia',aec=1
;ssw_cutout_service,'2011/06/07 06:20:00','2011/06/07 06:30:00',ref_helio='S22W53',fovx=350,fovy=350,email='guastavino@dima.unige.it',waves=[131,193],max_frames=1000,instrument='aia',aec=1
;
; NOTE: in the code images will be resized to 499 * 499 pixels FOV by ctrl_data function.
; So, a bigger FOV for the input data is needed.


add_path, './', /EXPAND
RESOLVE_ALL, /CONTINUE_ON_ERROR


tstart =   '10-sep-2017 16:06:08' ; start time selection  
tend   =   '10-sep-2017 16:06:10' ; end time selection  

cd, current=current_dir
path = current_dir+'\10sep2017'                    ; data storage folder path

;wav  = ['94','131','171','193','211','304','335']  ; wavelength to process
wav = ['171']

obj = obj_new('desat_pril')

result = obj ->desaturation( wav , tstart , tend  , path , save_fts = 1 , aec = 1, loud = 1) 
end
