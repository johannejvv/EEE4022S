
p1_pwr = 1.882240962021506e+05;
p1_sd = 2.442305698841263e+05;
p1_cov = p1_sd/p1_pwr;

p2_pwr =6.677687636639211e+04;
p2_sd = 1.251054440247152e+05;
p2_cov = p2_sd/p2_pwr;


p3_pwr =5.761053047715530e+05;
p3_sd = 1.032619068632877e+06;
p3_cov = p3_sd/p3_pwr;

p4_pwr = 2.568343632715954e+05;
p4_sd = 4.476781849106472e+05;
p4_cov = p4_sd/p4_pwr;

group_pwr = [p1_pwr p2_pwr p3_pwr p4_pwr]';
group_mean_pwr = mean(group_pwr);
group_sd = std(group_pwr);

x=1:5;
pwr_data = [p1_pwr p2_pwr p3_pwr p4_pwr group_mean_pwr]'./1000;

bar(x, pwr_data)
names = {'1'; '2'; '3'; '4'; 'group'};
set(gca,'xtick',[1:5],'xticklabel',names);
title('Average power in acceleration signal')
xlabel('Climbers')
ylabel('Average power [per 1000 Watts]')

y=1:4;
cov_data = [p1_cov p2_cov p3_cov p4_cov]';

figure
bar(y, cov_data)
title('Coefficient of Variance - acceleration power')
xlabel('Climbers')
ylabel('Standard deviation/mean')