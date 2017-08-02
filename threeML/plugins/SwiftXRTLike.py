from threeML.plugins.OGIPLike import OGIPLike

__instrument_name = "Swift XRT"


# At the moment this is just another name for the GenericOGIPLike spectrum
class SwiftXRTLike(OGIPLike):

    @staticmethod
    def _get_diff_flux_and_integral(likelihood_model):

        # In the XRT response matrix there are many many channels, so we can
        # use a very crude formula for the integral. This is why we override this

        n_point_sources = likelihood_model.get_number_of_point_sources()

        # Make a function which will stack all point sources (OGIP do not support spatial dimension)

        def differential_flux(energies):

            fluxes = likelihood_model.get_point_source_fluxes(0, energies)

            # If we have only one point source, this will never be executed
            for i in range(1, n_point_sources):
                fluxes += likelihood_model.get_point_source_fluxes(i, energies)

            return fluxes

        # The following integrates the diffFlux function using Simpson's rule
        # This assume that the intervals e1,e2 are all small, which is guaranteed
        # for any reasonable response matrix, given that e1 and e2 are Monte-Carlo
        # energies. It also assumes that the function is smooth in the interval
        # e1 - e2 and twice-differentiable, again reasonable on small intervals for
        # decent models. It might fail for models with too sharp features, smaller
        # than the size of the monte carlo interval.

        def integral(e1, e2):

            avg = (e2 + e1) / 2.0

            return differential_flux(avg) * (e2 - e1)

        return differential_flux, integral
