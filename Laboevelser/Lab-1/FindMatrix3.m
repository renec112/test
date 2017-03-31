function [n, Vreturn] = FindMatrix3(filnavn,V0)
% FINDMATRIX2 - returnerer en matrix V med potentialer fra en billedfil.
% Billedfilen skal v�re i bmp-format, og de farver der m� bruges er:
% R�d (255,0,0) => V=1
% Gr�n (0,255,0) => V=0.5
% Bl� (0,0,255) => V=-1
% Hvid (255,255,255) => V=0
% Der skal v�re et hvidt, r�dt og bl�t omr�de (hvis der ikke er et bl�t
% omr�de f�r kanten potentialet V=-1).

% Indl�s billedet.
billede = imread(filnavn,'bmp');

% St�rrelse af billedet
[n,m] = size(billede(:,:,1));

% Index i billedet hvor der er bl�t, gr�nt og r�dt (RGB).
red   = billede(:,:,1)==255 & ...
        billede(:,:,2)==0   & ...
        billede(:,:,3)==0;

green = billede(:,:,1)==0   & ...
        billede(:,:,2)==255 & ...
        billede(:,:,3)==0;
    
blue  = billede(:,:,1)==0 & ...
        billede(:,:,2)==0 & ...
        billede(:,:,3)==255;
  
  % S�t potentialet p� det bl�, gr�nne og r�de omr�de.
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

