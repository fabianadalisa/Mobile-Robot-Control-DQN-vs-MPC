function x_dot= my_pitch(t,x,u)
% cinematica robot differenziale
% stato x= [x_pos;y_pos; theta]

v = u(1);
w = u(2);

x_dot = [v*cos(x(3)); v*sin(x(3)) ; w ];

end