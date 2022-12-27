# Introduction
Before I'm finished with this project I would like to continue adjusting the code to try some edge cases where the points do not converge to the global minimum of the Riesz Energy. This project aims to test a projected gradient descent method on specific examples which are conjectured to converge towards a saddle point rather than the minimizer of the Riesz Energy for the Thompson Problem.

Included in this project are select examples for n=4 to compare the edge cases to see which converge to the known minimizer, being the tetrahedron, against the only conjectured saddle point, the square whose vertices are located on the equator (up to invariance under rotation). Some further examples I would like to eventually include are n=5 which hopefully are extrapolated from the n=4 cases.

Below is the example of a kite configuration, interestingly this converges to the minimum.
![](https://github.com/edawhite/RieszEnergy/blob/main/kite.gif)

Contrast this with the example of a coplanar rectangular configuration, which unsurprisingly converges to the saddle point
![](https://github.com/edawhite/RieszEnergy/blob/main/rectangle.gif)

## Summary so far
Viewing the gif files rectangle, quadrilateral, and kite we notice that the rectangle converges to the saddle point of the Riesz Energy, which is to be expected as it is very similar to the saddle points of a square on any great circle.
Interestingly both the quadrilateral and kite appear to initially converge to the saddle point, however eventual become perturbed enough to destabalize toward the global minimum. This seems to suggest a degree of symmetry is necessary to be an exceptional case to converge to the saddle point and otherwise we can expect to converge to the global minimum for 4 points.
Finally, the testing gif file shows an example of what happens when given a random collection of 4 points. In all, these examples seem to suggest that certain cases which exhibit a high degree of symmetry seem to converge to the saddle point, where configurations with fewer or even no symmetries converge to the minimizer. This will be important to consider as we look to configurations of more points. (n>4)
