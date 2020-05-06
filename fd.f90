program fd
    implicit none
    
    double precision, dimension(10000,10000) :: a, b
    double precision :: h
    integer :: i, j, n

    a=0.0d0 

    ! set the static boundary conditions

    do i=1,10000
        a(i,1)=1.0d0
        a(i,10000)=2.0d0
        a(1,i)=3.0d0
        a(10000,i)=4.0d0
    end do

    b=a

    ! Run through the matrix solving for the stencil
    do n=1, 10
        do i=2, 9999
            do j=2, 9999
                b(i,j)=a(i,j)/2.0d0+a(i+1,j)/8.0d0 + a(i-1,j)/8.0d0 + a(i,j+1)/8.0d0 + a(i,j-1)/8.0d0
            end do
        end do
        a=b
    end do

    write(*,*) a(5,5),a(1000,1000),a(9995,9995)

end program
