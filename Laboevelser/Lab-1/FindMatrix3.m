function [n, Vreturn] = FindMatrix3(filnavn,V0)
% FINDMATRIX2 - returnerer en matrix V med potentialer fra en billedfil.
% Billedfilen skal være i bmp-format, og de farver der må bruges er:
% Rød (255,0,0) => V=1
% Grøn (0,255,0) => V=0.5
% Blå (0,0,255) => V=-1
% Hvid (255,255,255) => V=0
% Der skal være et hvidt, rødt og blåt område (hvis der ikke er et blåt
% område får kanten potentialet V=-1).

% Indlæs billedet.
billede = imread(filnavn,'bmp');

% Størrelse af billedet
[n,m] = size(billede(:,:,1));

% Index i billedet hvor der er blåt, grønt og rødt (RGB).
red   = billede(:,:,1)==255 & ...
        billede(:,:,2)==0   & ...
        billede(:,:,3)==0;

green = billede(:,:,1)==0   & ...
        billede(:,:,2)==255 & ...
        billede(:,:,3)==0;
    
blue  = billede(:,:,1)==0 & ...
        billede(:,:,2)==0 & ...
        billede(:,:,3)==255;
  
  % Sæt potentialet på det blå, grønne og røde område.
  V = zeros(n);
  if any(any(blue))
    V(blue) = -V0;
  else
    V(1,:) = -V0;
    V(n,:) = -V0;
    V(:,1) = -V0;
    V(:,m) = -V0;
  end
  V(red) = V0;
  if any(any(green))
    V(green) = V0/2;
  end
  Vreturn = V;

