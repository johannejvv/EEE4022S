 
COM_x = table2array(p4COM(:,1));
COM_y = table2array(p4COM(:,2));
time = [0: 1/30 : (length(COM_x)-1)*(1/30)]';
scatter(COM_x, COM_y)
set(gca, 'YDir','reverse')
figure

plot(time, COM_y)
set(gca, 'YDir','reverse')
title('Y position of CoM')
xlabel('Time (s)')
ylabel('Position (pixels)')
figure

plot(time, COM_x)
figure

t = 1:100 ;
vel_x = zeros(length(time)-1,1) ;
vel_y = zeros(length(time)-1,1) ;
for i = 2:length(time)-1
    vel_x(i) = (COM_x(i)-COM_x(i-1))/(time(i)-time(i-1)) ;
    vel_y(i) = (COM_y(i)-COM_y(i-1))/(time(i)-time(i-1)) ;
end

plot(time(2:end), vel_y)
title('y velocity CoM')
xlabel('Time (s)')
ylabel('velocity (pixels)')
figure

plot(time(2:end), vel_x)
title('x velocity CoM')
xlabel('Time (s)')
ylabel('velocity (pixels)')
figure

vel = sqrt(vel_x.^2 + vel_y.^2);
mean_vel = mean(vel);
SD_vel = std(vel);

 plot(time(1:length(vel)), vel);
title('Absolute velocity of CoM')
xlabel('Time (s)')
ylabel('Absolute velocity (pixels/s)')
figure

acc_x = zeros(length(time)-2,1) ;
acc_y = zeros(length(time)-2,1) ;
for i = 2:length(time)-2
    acc_x(i) = (vel_x(i)-vel_x(i-1))/(time(i)-time(i-1)) ;
    acc_y(i) = (vel_y(i)-vel_y(i-1))/(time(i)-time(i-1)) ;
end
 
acc = sqrt(acc_x.^2 + acc_y.^2);
 
mean_acc = mean(acc);
SD = std(acc);
plot(time(1:length(acc)),acc);
title('Absolute acceleration of CoM')
xlabel('Time (s)')
ylabel('Absolute acceleration (pixels/s^2)')
figure

Fs=30;
N = length(acc);
xdft = fft(acc);
xdft = xdft(1:N/2+1);
psdx = (1/(Fs*N)) * abs(xdft).^2;
psdx(2:end-1) = 2*psdx(2:end-1);
freq = 0:Fs/length(acc):Fs/2;

plot(freq,10*log10(psdx))
grid on
title('Power Spectral Density')
xlabel('Frequency (Hz)')
ylabel('Power/Frequency (dB/Hz)')

delta_f = Fs/length(acc);

total_power = sum(psdx(2:end))*delta_f;

power = zeros(length(psdx)-1,1) ;
for i = 2:length(power)-1
    power(i) = psdx(i)*delta_f;
end

mean_power = mean(power);
sd_power =std(power);

freq_bands = 0.1:0.1:14;
percent_power = (power*100/total_power);
ratio_power = power/total_power;

M = movmean(percent_power,[9 0]);
C = cumsum(ratio_power);

figure
plot(freq(2:end),percent_power); hold on; grid on;
plot(freq(2:end),M); hold on; grid on;
plot(freq(2:end),C);
title('Normalized power in acceleration signal')
ylabel('% Power')
xlabel('Frequency (Hz)')
legend('percentage power','moving average', 'cumulative sum');





