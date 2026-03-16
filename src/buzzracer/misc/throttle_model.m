wR_sim = zeros(1,length(wR));
wRi = wR(1);
for ii = 1:length(wR)
    wR_sim(ii) = wRi;
    wRi = update(wRi, throttle(ii));
end
figure;
plot(1:length(wR), wR);
hold on;
plot(1:length(wR), wR_sim);

function wR_new = update(wRi, T)
    A = -1;
    B0 = -20;
    B1 = 700;
    B2 = -450;
    wR_new = wRi + (A*wRi + B0 + B1*T + B2*T^2);
end
    