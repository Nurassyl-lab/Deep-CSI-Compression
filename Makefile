# from me to ubuntu server
sync_dry:
	rsync -avz --dry-run --exclude-from=exclude.txt ./ lab716a_ubu3060:~/Documents/snap_csi_compression/

sync:
	rsync -avz --exclude-from=exclude.txt ./ lab716a_ubu3060:~/Documents/snap_csi_compression/

sync_delete_dry:
	rsync -avz --dry-run --delete --exclude-from=exclude.txt ./ lab716a_ubu3060:~/Documents/snap_csi_compression/

sync_delete:
	rsync -avz --delete --exclude-from=exclude.txt ./ lab716a_ubu3060:~/Documents/snap_csi_compression/


# from ubuntu server to me
sync_back_dry:
	rsync -avz --dry-run --exclude-from=exclude.txt lab716a_ubu3060:~/Documents/snap_csi_compression/ ./

sync_back:
	rsync -avz --exclude-from=exclude.txt lab716a_ubu3060:~/Documents/snap_csi_compression/ ./

