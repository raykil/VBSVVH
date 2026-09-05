from coffea.nanoevents.schemas.nanoaod import NanoAODSchema


class RDFSchema(NanoAODSchema):
    """Custom schema derived from NanoAODSchema to handle RDF output

    This inherits all NanoAODSchema functionality but overrides mixins for specific collections.
    All other collections will use NanoAODSchema's default handling (e.g., 'NanoCollection' fallback).
    Cross-references, nested items, etc., are preserved from the base class.

    To disable warnings for missing cross-references (if your files don't have them):
    Set warn_missing_crossrefs = False.

    To use: In your coffea processor or factory, pass schemaclass=CustomSchema.
    """

    warn_missing_crossrefs = False  # Optional: Disable warnings if not needed

    mixins = {
        **NanoAODSchema.mixins,  # Inherit base mixins and override/add as needed
        "electron": "PtEtaPhiMCollection",
        "muon": "PtEtaPhiMCollection",
        "lepton": "PtEtaPhiMCollection",
        "jet": "PtEtaPhiMCollection",
        "fatjet": "PtEtaPhiMCollection",
        "met": "MissingET",
        "gen_h": "PtEtaPhiMCollection",
        "gen_b1": "PtEtaPhiMCollection",
        "gen_b2": "PtEtaPhiMCollection",
        "gen_v1": "PtEtaPhiMCollection",
        "gen_v1q1": "PtEtaPhiMCollection",
        "gen_v1q2": "PtEtaPhiMCollection",
        "gen_v2": "PtEtaPhiMCollection",
        "gen_v2q1": "PtEtaPhiMCollection",
        "gen_v2q2": "PtEtaPhiMCollection",
        "gen_vbs1": "PtEtaPhiMCollection",
        "gen_vbs2": "PtEtaPhiMCollection",
    }