% Programmet udregner og plotter �kvipotentialkurver og E-felt fra
% ledere. Konfigurationen af lederne kan indtastes direkte i en matrix eller
% indl�ses fra en bitmap-fil. P� det r�de omr�de s�ttes potentialet til +V0,
% p� det bl� til -V0, p� det gr�nne til V0/2 og p� det hvide til 0. Er der
% intet bl�t omr�de s�ttes kanten til -V0, ellers bestemmes potentialet p�
% kanten som gennemsnittet af potentialet fra nabofelterne (ligesom resten
% af det hvide omr�de). Potentialet skal v�re det samme overalt p� det
% gr�nne omr�de, men v�rdien kendes ikke fra starten, s� der skal ogs�
% itereres over dette omr�de.

% Brug kvadratisk billede!!!


clear all; clc; close all

% V�rdi af potentialet p� det r�de omr�de.
V0 = 1;

% Indtast v�rdien af potentialerne i en matrix...
% $$$ m = 50;
% $$$ Vinit = -ones(m,m);
% $$$ Vinit(1,:) = V0;
% $$$ %Vinit(m,:) = V0;
% $$$ Vinit(2:m-1,2:m-1) = 0;


% ...eller indl�s billedet fra en fil.
filnavn = 'Eksempel_elektrodekonfigurationer/ring1.bmp';
[n, Vinit] = FindMatrix3(filnavn,V0);

% Vis evt. billedet.
figure(1);
billede = imread(filnavn,'bmp');
image(billede);
set(gca,'YDir','normal');
axis image;

% Gem den matrix der startes med.
V = Vinit;

% Mindste v�rdi af potentialet.
Vmin = min(min(V));

% N�jagtighed.
epsilon = 0.001;

% St�rrelse af �ndringen af potentialet
delta = inf;

% Index i billedet hvor der er bl�t, gr�nt og r�dt.
red     = Vinit ==  V0;
[Ired,Jred] = find(red);

blue    = Vinit == -V0;
[Iblue,Jblue] = find(blue);

green   = Vinit ==  V0/2;
[Igreen,Jgreen] = find(green);

white   = Vinit ==  0;
target  = green | white;

% Find naboer
naboMatrix = [zeros(n-1,1) eye(n-1); zeros(1, n)]...
              + [zeros(1, n); eye(n-1) zeros(n-1,1)];

antalNaboer = naboMatrix*ones(n)+(naboMatrix*ones(n)')';
            
% Bliv ved med at iterere indtil n�jagtigheden er stor nok.
while delta>epsilon,
    % Gem den tidligere matrix som reference
    Vold = V;
  
    omkredsPotentiale = naboMatrix*V+(naboMatrix*V')';
  
    V(target) = omkredsPotentiale(target)./antalNaboer(target);
  
    % Udregn gennemsnittet af det gr�nne omr�de og s�t potentialet p� dette
    % omr�de lig med gennemsnittet.
    if any(any(green))
        V(green) = mean(mean(V(green)));
    end
  
    % Beregn st�rrelsen af �ndringen.
    % delta = max(max(abs(V-Vold)));
    delta = max(max(abs((V-Vold)./(V-Vmin+eps))));
end

% Beregn E-feltet.
[DVx,DVy] = gradient(-V);

figure(2);
imagesc(V);
% $$$ hold on;
% $$$ plot(Jblue,Iblue,'wx');
% $$$ plot(Jred,Ired,'wx');
% $$$ plot(Jgreen,Igreen,'wx');
% $$$ hold off;
colorbar;
set(gca,'YDir','normal');
axis square;

figure(3);
contour(V,20);  % Plot potentialkurver.
hold on;
quiver(DVx,DVy,2,'k');  % Plot E-feltet.
plot(Jred,Ired,'rs','MarkerFaceColor','r',...
                'MarkerSize',4);
plot(Jgreen,Igreen,'gs','MarkerFaceColor','g',...
                'MarkerSize',4);
plot(Jblue,Iblue,'bs','MarkerFaceColor','b',...
                'MarkerSize',4);
hold off;
colorbar;
axis square;
