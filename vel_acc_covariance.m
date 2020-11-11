
p1_acc =8.687771191792483e+03;
p1_sd = 6.756496561809870e+03;
p1_vel = 3.443833558218680e+02;
p1_sd_vel = 2.434503618833346e+02;
p1_cov = p1_sd/p1_acc;
p1_cov_vel = p1_sd_vel/p1_vel;


p2_acc =6.969212661320472e+03;
p2_sd = 6.975377913767085e+03;
p2_vel = 1.789656709774188e+02;
p2_sd_vel = 1.673988781411599e+02;
p2_cov = p2_sd/p2_acc;
p2_cov_vel = p2_sd_vel/p2_vel;

p3_acc =1.112789562817207e+04;
p3_sd = 1.435419072303871e+04;
p3_vel = 2.852228451112401e+02;
p3_sd_vel =3.845941706690444e+02;
p3_cov = p3_sd/p3_acc;
p3_cov_vel = p3_sd_vel/p3_vel;

p4_acc = 1.027200729374447e+04;
p4_sd = 1.055381106470932e+04;
p4_vel = 2.447754461815388e+02;
p4_sd_vel = 2.616311267798075e+02;
p4_cov = p4_sd/p4_acc;
p4_cov_vel = p4_sd_vel/p4_vel;

group_acc = [p1_acc p2_acc p3_acc p4_acc]';
group_mean_acc = mean(group_acc);
group_sd = std(group_acc);

group_vel = [p1_vel p2_vel p3_vel p4_vel]';
group_mean_vel = mean(group_vel);
group_vel_sd = std(group_vel);

x=1:5;
acc_data = [p1_acc p2_acc p3_acc p4_acc group_mean_acc]'./1000;
vel_data = [p1_vel p2_vel p3_vel p4_vel group_mean_vel]'./1000;

bar(x, acc_data)
names = {'1'; '2'; '3'; '4'; 'group'};
set(gca,'xtick',[1:5],'xticklabel',names);
title('Average acceleration')
xlabel('Climbers')
ylabel('Average acceleration [1000pixels/s^2]')

figure
bar(x, vel_data)
names = {'1'; '2'; '3'; '4'; 'group'};
set(gca,'xtick',[1:5],'xticklabel',names);
title('Average velocity')
xlabel('Climbers')
ylabel('Average velocity [1000pixels/s]')


y=1:4;
cov_data = [p1_cov p2_cov p3_cov p4_cov]';
cov_data_vel = [p1_cov_vel p2_cov_vel p3_cov_vel p4_cov_vel]';

figure
bar(y, cov_data)
title('Coefficient of Variance - acceleration')
xlabel('Climbers')
ylabel('Standard deviation/mean')

figure
bar(y, cov_data_vel)
title('Coefficient of Variance - velocity')
xlabel('Climbers')
ylabel('Standard deviation/mean')

