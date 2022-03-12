c
      include 'commons_panel'
c
      pi = 3.1415926
c
      call read_data
c
      call set_target_points

      call solve(1)

      call solve(2)

      call input_data_and_plot

      stop
      end


c
c------------------Input data and plot-----------------------------------------
c
      subroutine input_data_and_plot

      character*1,answer

      include 'commons_panel'

888   print*,'Enter inlet velocity (magnitude, angle(deg))'
      read*,Vmag,Vangle
      print*,Vmag,Vangle

      call result_plot
      
      print*,'Would you like to try another inlet condition? (y/n)'
      read*,answer
      if (answer.eq.'y') then 
        goto 888
      end if

      return
      end
      
c
c---------------solve--------------------------------------------
c
      subroutine solve(gamindx)
      include 'commons_panel'
      integer gamindx
      if (gamindx.eq.1) then
         U0=1
         V0=0
      else
         U0=0
         V0=1
      end if

      print*,gamindx,U0,V0
c
      call set_uind
c
      call set_influence_coefficients
c
      call apply_kutta
c
      call invert(gamindx)

      return
      end
c
c-------------result plot----------------------------------------------
c
      subroutine result_plot

      include 'commons_panel'

     

      Vact1=-(gamtot(1)/(2*a))
      Vact2=1-(gamtot(2)/(2*a))

      print*,gamtot(1),gamtot(2)
      print*,Vact1,Vact2
      do i=1,ntot
      print*,i,gam(i,1),gam(i,2)
      end do
      print*,Vmag,Vangle
      Ureq=Vmag*cos(Vangle*2*pi/360)
      Vreq=Vmag*sin(Vangle*2*pi/360)

      coeff1=Ureq
      coeff2=(Vreq-Vact1*Ureq)/Vact2

      print*,'Ureq = ',Ureq,' Vreq = ',Vreq
      print*,coeff1,coeff1*Vact1+coeff2*Vact2


      do i=1, ntot
        gamma(i)=coeff1*gam(i,1) + coeff2*gam(i,2)
        gv(i)=gamma(i)/dl(i)
      end do

      call calc_surf_vels

      call plot_shape

      call streamlines

      return
      end
c
c---------------read data--------------------------------------------
c
      subroutine read_data

      include 'commons_panel'
  
      open (unit=1, file='shapenew')

      read (1,*) ntot
      read (1,*) ikutta
      print *, ntot

      do i=1, ntot
         read (1,*)  xv(i), yv(i)
c        read (1,*) xc(i), xv(i), yv(i)
c       print*,i,xv(i),yv(i)
      end do

      read (1,*) xmin, xmax, ymin, ymax, nhts
      print*,nhts
c
      return
      end
c
c----------------------set target points-------------------------------------
c
      subroutine set_target_points
c
      include 'commons_panel'
c
c set target points (midway between vortices) 
c positions given by (xt,yt)
c normals (nhatx,nhaty) (positive outward)
c and dl's (panel lengths)
c
      ntotm1=ntot         
      do i=1, ntotm1
        ip1=i+1

        if (i.EQ.ntot) then
          ip1=1
        end if

        xt(i)=(xv(i)+xv(ip1))/2
        yt(i)=(yv(i)+yv(ip1))/2
        dx=xv(ip1)-xv(i)
        dy=yv(ip1)-yv(i)     
        dl(i)=sqrt(dx*dx + dy*dy)
        nhatx(i)=dy/dl(i)
        nhaty(i)=-dx/dl(i)


      end do

      return
      end
c
c------------set uind-----------------------------------------------
c
      subroutine set_uind
c
      include 'commons_panel'
      print*,U0,V0
c
c
c calculate velocity components induced at each target point
c
        ntotm1=ntot -1
	do i=1,ntot
c
c          due to free stream
c
          uindx(i) = U0
   	  uindy(i) = V0
c
c           now take normal component
c
	  uind(i) =  uindx(i)*nhatx(i) + uindy(i)*nhaty(i)
c
	end do
        uind(ntot)=0.0
c
      return
      end
c
c-----------set influence coefficients------------------------------------------------
c
      subroutine set_influence_coefficients
c
      include 'commons_panel'
      print*,U0,V0
c
c
c calculate influence coefficients for flow induced by vortices
c   (   ainf(i,j)*gam(j) is the flow normal to the i'th target point due to
c       the j'th vortex )
c
      ntotm1=ntot-1
      do i=1, ntotm1
        do j=1, ntot
          x=(2*pi*(xt(i)-xv(j)))/a
          y=(2*pi*(yt(i)-yv(j)))/a
          ainf(i,j) = -(nhatx(i)*(1/(2*a))*sin(y))/(cosh(x)-cos(y))
     &                +(nhaty(i)*(1/(2*a))*sinh(x))/(cosh(x)-cos(y))
c        print *,i,j,ainf(i,j)
        
        end do
      end do

c
      return
      end
c
c------------apply kutta-----------------------------------------------
c
      subroutine apply_kutta
c
      include 'commons_panel'
c
c
c final equation is replaced by Kutta condition
c that gv(1) = -gv(ntot)
c
      print*,'ikutta = ',ikutta
      ikuttam1=ikutta - 1
      if (ikutta.EQ.1) then
        ikuttam1=ntot
      end if
      
      do i=1, ntot
        if (i.EQ.ikutta .OR. i.EQ.ikuttam1) then
          ainf(ntot,i)=1/dl(i)
        else
          ainf(ntot,i)=0.0
        end if
      end do

c      do i=1,ntot
c        print*,i,ainf(ntot,i)
c      end do


c
      return
      end
c
c------------invert-----------------------------------------------
c
      subroutine invert(gamindx)
      integer gamindx
c
      include 'commons_panel'
c
c solve matrix of equations to find vortex strengths
c This is 
c        ainf(i,j) * gam(j) + uind(i) = 0
c
c declare extra workspace
c
      dimension wk(nmax_vort,nwks), cc(nmax_vort), ip(nmax_vort)
c
      do i=1,ntot
        ip(i) = i
      end do
c
      do i=1,ntot
        do j=1,ntot
          wk(i,j) = ainf(i,j)
        end do
        wk(i,ntot+1) = -uind(i)
      end do
c
      do k=1,ntot
        tl = 0.
        lm = 0
        do l=k,ntot
          itemp = ip(l)
          te = abs(wk(itemp,k))
          if(te.ge.tl) then
            lm = l
            tl = te
          end if
        end do
c     
        itemp = ip(lm)
        ip(lm) = ip(k)
        ip(k) = itemp
        d = 1./wk(itemp,k)
        do i=1,ntot
          cc(i) = wk(i,k)*d
        end do
        do i=1,ntot
          if( i.ne.itemp ) then
            do j=k+1,ntot+1
              wk(i,j) = wk(i,j)-cc(i)*wk(itemp,j)
            end do
          end if
        end do
        do j=k+1,ntot+1
          wk(itemp,j) = wk(itemp,j)*d
        end do
      end do
c
c      print*,'gamindx is ',gamindx
      gamtot(gamindx)=0
c      print*,'gamtot (',gamindx,') is ',gamtot(gamindx)
      do i=1,ntot
        gam(i,gamindx) = wk(ip(i),ntot+1)
        gamtot(gamindx)=gamtot(gamindx)+gam(i,gamindx)
      end do
c      print*,'gamtot (',gamindx,') is ',gamtot(gamindx)
c      print*,'gamtot(1)=',gamtot(1)
c      print*,'gamtot(2)=',gamtot(2)
c
      return
      end
c
c------------plot shape-----------------------------------------------
c
      subroutine plot_shape
c
      include 'commons_panel'
c
      call selplt(4)
      call origin(30.,30.,0)
      xscale = 210./(xmax-xmin)
      yscale = 150./(ymax-ymin)
      if( xscale.ge.ysale ) then
        call scale( yscale, yscale)
      else
        call scale( xscale, xscale )
      end if
      call origin( -xmin, -ymin )
c
     
      do j=-1,2
        call moveto( xv(1), yv(1)+j*a)
      do i=1,ntot
        ip1=i+1
        if (i.EQ.ntot) then
          ip1=1
        end if
        call drawto( xv(ip1), yv(ip1)+j*a)
      end do
      end do
      
c
      return
      end
c
c------------stream lines-----------------------------------------------
c
      subroutine streamlines
   
c
      parameter (ngrid=100, nhts_max=100)
c
      include 'commons_panel'
c
      dimension xg(ngrid,ngrid), yg(ngrid,ngrid), 
     &          psival(ngrid,ngrid), hts(nhts_max),
     &          w1(ngrid,ngrid), w2(ngrid,ngrid)
c

 
  100   format(' *** no of contours too large ***',
     &         ' Value requested = ',i5,' Max = ',i5)
    
      
c
      psimin =  1000000.
      psimax = -1000000.
      do i=1,ngrid
        do j=1,ngrid
          xg(i,j) = xmin + (i-1)*(xmax-xmin)/float(ngrid-1)
          yg(i,j) = ymin + (j-1)*(ymax-ymin)/float(ngrid-1)
          psival(i,j) = psi(xg(i,j),yg(i,j))
          if( psival(i,j).lt.psimin ) psimin = psival(i,j)
          if( psival(i,j).gt.psimax ) psimax = psival(i,j)
        end do
      end do
c
      do n=1,nhts
        hts(n) = psimin + (n-1)*(psimax-psimin)/float(nhts-1)
      end do
c
      call contor(xg,yg,psival,ngrid,ngrid,ngrid,hts,nhts,0,w1,w2)
      call brkplt
c
      return
      end
c
c------------psi---------------------------------------------------------
c
      function psi(x,y)
c
      include 'commons_panel'
c
c  calculate stream function for the total flow
c
c    first due to free stream
c
      psi = coeff1*y - coeff2*x
c
c    now due to vortices on boundary
c
      do i=1,ntot
        xa=(2*pi*(x-xv(i)))/a
        ya=(2*pi*(y-yv(i)))/a
         
        psi = psi - (gamma(i)/(4*pi))*log(0.5*(cosh(xa)-cos(ya)))
       
      end do
c
      return
      end

c
c--------------salc surf vels-------------------------------------------------------
c
      subroutine calc_surf_vels

      include 'commons_panel'

      open (unit=3, file='vels')
 
      write(3,*) ntot, a, Vmag, Vangle
      write(3,*) xmin, xmax, ymin, ymax
    
      do i=1,ntot
        print*,i,xc(i),gv(i)
        write(3,*) xc(i),abs(gv(i)/(2*Vmag)),xv(i),yv(i),gamma(i)
      end do
 
      close (3)
      return
      end