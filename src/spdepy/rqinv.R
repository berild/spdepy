#
#rqinv <- function(tfile){
#  tmp = RcppCNPy::npyLoad(tfile)
  #tmp_toInla = tempfile()
#  Q = sparseMatrix(i = tmp[,1]+1,j = tmp[,2]+1,x=tmp[,3])
#  Q = INLA::inla.as.dgTMatrix(Q)
  #write(t(cbind(Q@i,  Q@j,  Q@x)), ncol=3, file=tmp_toInla)
#  Qinv = INLA::inla.qinv(Q)
  #exeName = "~/Library/R/4.0/library/INLA/bin/mac/64bit/inla.run"
  #com = sprintf('%s -s -m qinv %s %s %s', exeName, tmp_toInla, tmp_toInla, tmp_fromInla)
  #res = system(com)
  #unlink(tmp_toInla)
#  res = cbind(Qinv@i,Qinv@j,Qinv@x)
#  RcppCNPy::npySave(sprintf("%sfromINLA.npy",strsplit(tfile,".npy")[[1]]),res)
#  return(T)
#  return(res)
options(warn=-1)
rqinv = function(A)
{
  INLA::inla.setOption("smtp" = "pardiso", pardiso.license = paste0(getwd(),"/pardiso.lic"))
  version = 0
  #tmpdir = "/cluster/home/martinob/R/tmp"
  tmpdir = tempdir()
  filename = tempfile() #paste0(tmpdir, "/in")
  A = INLA::inla.as.dgTMatrix(A)
  nrow = dim(A)[1]
  ncol = dim(A)[2]
  datatype = 1 ## sparse
  valuetype = ifelse(is.integer(A), integer(), double())
  matrixtype = 0  ## general
  storagetype = 1 ## columnmajor
  i = A@i
  j = A@j
  values = A@x
  elems = length(i)
  h = integer(8)
  valuetp = ifelse(identical(valuetype, integer()), 0, 1)
  h = c(version, elems, nrow, ncol, datatype, valuetp, matrixtype, storagetype)
  
  fp = file(filename, "wb")
  writeBin(as.integer(length(h)), fp)
  writeBin(as.integer(h), fp)
  
  if (datatype == 0) {
    ## dense
    if (identical(valuetype, integer())) {
      writeBin(as.integer(as.vector(A)), fp)
    } else {
      writeBin(as.double(as.vector(A)), fp)
    }
  } else {
    writeBin(as.integer(i), fp)
    writeBin(as.integer(j), fp)
    if (identical(valuetype, integer())) {
      writeBin(as.integer(values), fp)
    } else {
      writeBin(as.double(values), fp)
    }
  }
  close(fp)
  constr.file = tempfile(tmpdir = tmpdir)# paste0(tmpdir,"/constr") 
  out.file = tempfile(tmpdir = tmpdir) # paste0(tmpdir,"/out") 
  if  (Sys.info()['sysname']=="Linux"){
    where = paste(system.file(package = 'INLA'),'/bin/linux/64bit/inla.run',sep="")
    #where = "/cluster/home/martinob/R/x86_64-pc-linux-gnu-library/4.1/INLA/bin/linux/64bit/inla.run"
  }else{#/Library/Frameworks/R.framework/Versions/4.0/Resources/library/INLA/bin/linux/64bit
    #where = "~/Library/R/4.0/library/INLA/bin/linux/64bit/inla.run" 
    where = paste(system.file(package = 'INLA'),'/bin/mac/64bit/inla.run',sep="")
  }
  system(paste(where, "-s -m qinv", filename, constr.file, out.file))

  fp = file(out.file, "rb")
  
  len.h = readBin(fp, what = integer(), n = 1)
  ## currently required
  stopifnot(len.h >= 8)
  
  h.raw = readBin(fp, what = integer(), n = len.h)
  
  h = list(filename = out.file,
           version = h.raw[1],
           elems = h.raw[2],
           nrow = h.raw[3],
           ncol = h.raw[4],
           datatype = ifelse(h.raw[5] == 0, "dense", "sparse"),
           valuetype = ifelse(h.raw[6] == 0, integer(), double()),
           matrixtype = ifelse(h.raw[7] == 0, "general",
                                    ifelse(h.raw[7] == 1, "symmetric", "diagonal")),
           storagetype = ifelse(h.raw[8] == 0, "rowmajor", "columnmajor"))
  
  if (h$datatype == "dense") {
    ##
    ## dense matrix
    ##
    if (h$matrixtype != "general")
      stop(paste("Read", filename, ". Type (`dense' && !`general') is not yet implemented."))
    
    stopifnot(h$elems == h$nrow * h$ncol)
    Aelm = readBin(fp, what = h$valuetype, n = h$elems)
    read.check(Aelm, h)
    A = matrix(Aelm, nrow = h$nrow, ncol = h$ncol, byrow = (h$storagetype == "rowmajor"))
  } else if (h$datatype == "sparse") {
    ##
    ## sparse matrix
    ##
    if (h$storagetype == "rowmajor") {
      ##
      ## rowmajor format
      ##
      i = c()
      j = c()
      values = c()
      if (h$matrixtype == "symmetric") {
        ##
        ## symmetric
        ##
        for(k in 1:h$elems) {
          ij = readBin(fp, what = integer(), n = 2)
          i = c(i, max(ij))
          j = c(j, min(ij))
          values = c(values, readBin(fp, what = h$valuetype, n = 1))
        }
        read.check(i, h)
        read.check(j, h)
        read.check(values, h)
        
        ## oops. Matrix adds replicated elements!!!
        if (!(all(i >= j) || all(i <= j)))
          stop(paste("Reading file", filename,
                     ". Both upper and lower part of symmetric sparse matrix",
                     "is specified. Do not know what to do."))
        idx = (i != j)
        ii = i[idx]
        jj = j[idx]
        i = c(i, jj)
        j = c(j, ii)
        values = c(values, values[idx])
      } else if (h$matrixtype == "general" || h$matrixtype == "diagonal") {
        ##
        ## general/diagonal
        ##
        for(k in 1:h$elems) {
          ij = readBin(fp, what = integer(), n = 2)
          i = c(i, ij[1])
          j = c(j, ij[2])
          values = c(values, readBin(fp, what = h$valuetype, n = 1))
        }
        
        if (h$matrixtype == "diagonal") {
          idx = (i == j)
          i = i[idx]
          j = j[idx]
          values = values[idx]
        }
      } else {
        stop("This should not happen.")
      }
    } else {
      ##
      ## columnmajor format
      ##
      
      ##
      ## other format: (i, j, values)
      ##
      i = readBin(fp, what = integer(0), n = h$elems)
      j = readBin(fp, what = integer(0), n = h$elems)
      values = readBin(fp, what = h$valuetype, n = h$elems)
      
      if (h$matrixtype == "symmetric") {
        ##
        ## symmetric: lower or upper triangular part is given
        ##
        
        ## oops. Matrix adds replicated elements!!!
        if (!(all(i >= j) || all(i <= j)))
          stop(paste("Reading file", filename,
                     ". Both upper and lower part of symmetric sparse matrix",
                     "is specified. Do not know what to do..."))
        
        idx = (i != j)
        ii = i[idx]
        jj = j[idx]
        ## yes, this is correct...
        i = c(i, jj)
        j = c(j, ii)
        values = c(values, values[idx])
      } else if (h$matrixtype == "diagonal") {
        idx = (i == j)
        i = i[idx]
        j = j[idx]
        values = values[idx]
      }
    }
    A = sparseMatrix(i = i, j = j, x = values, dims = c(h$nrow, h$ncol),
                     index1 = FALSE)
    A = INLA::inla.as.dgTMatrix(A)
  } else {
    stop("This should not happen.")
  }
  close(fp)
  file.remove(filename)
  file.remove(out.file)
  return (cbind(A@i,A@j,A@x))
}



