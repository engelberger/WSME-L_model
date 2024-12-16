import numpy as np
from Bio.PDB import PDBParser, Selection
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler("log.txt", mode="w"),
        logging.StreamHandler(),  # This will still print to console
    ],
)
logger = logging.getLogger(__name__)


def load_reference_map(filename):
    """Load reference contact map for comparison"""
    try:
        return np.loadtxt(filename)
    except Exception as e:
        logger.error(f"Error loading reference file {filename}: {e}")
        return None


def log_pdb_info(structure, chain_id):
    """Log detailed information about the PDB structure"""
    model = structure[0]
    chain = model[chain_id]

    logger.info("\nPDB Structure Information:")
    logger.info("-" * 50)
    logger.info(f"Number of models: {len(structure)}")
    logger.info(f"Chains in model 0: {[chain.id for chain in model]}")
    logger.info(f"Selected chain: {chain_id}")

    # Count CA atoms and residues
    ca_residues = [res for res in chain if "CA" in res]
    logger.info(f"Number of residues with CA atoms: {len(ca_residues)}")
    logger.info(f"Residue numbers: {[res.get_id()[1] for res in ca_residues]}")
    logger.info("-" * 50 + "\n")


def generate_distance_contact_maps(
    pdb_file,
    output_dm,
    output_cm,
    n_residues=60,
    cutoff=8.0,
    contact_energy=-2.506,
    chain_id="A",
):
    """
    Generate distance map and contact map for src SH3 domain
    Parameters match those described in the paper
    """
    warnings.simplefilter("ignore", PDBConstructionWarning)

    # Parse PDB file and get structure info
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("SH3", pdb_file)
    log_pdb_info(structure, chain_id)

    # Get CA atoms
    ca_atoms = {}
    model = structure[0]
    chain = model[chain_id]

    # Map residue numbers to match expected numbering (1-60)
    residue_mapping = {}
    sorted_residues = sorted([res.get_id()[1] for res in chain if "CA" in res])
    for i, res_num in enumerate(sorted_residues, 1):
        residue_mapping[res_num] = i
        logger.info(f"Mapping residue {res_num} to position {i}")

    # Get CA atoms with mapped numbering
    for residue in chain:
        if "CA" in residue:
            orig_num = residue.get_id()[1]
            mapped_num = residue_mapping[orig_num]
            if 1 <= mapped_num <= n_residues:
                ca_atoms[mapped_num] = residue["CA"]

    # Initialize maps
    distance_map = np.zeros((n_residues, n_residues))
    contact_map = np.zeros((n_residues, n_residues))
    max_distance = 999.0

    # Calculate distances and contacts
    for i in range(1, n_residues + 1):
        for j in range(i, n_residues + 1):  # Only calculate upper triangle
            if i in ca_atoms and j in ca_atoms:
                coord_i = ca_atoms[i].get_coord()
                coord_j = ca_atoms[j].get_coord()
                distance = np.linalg.norm(coord_i - coord_j)

                # Store distances symmetrically
                distance_map[i - 1, j - 1] = distance
                distance_map[j - 1, i - 1] = distance

                # Calculate contact energy based on physical model from paper
                if i != j:  # Skip self-contacts
                    seq_sep = abs(i - j)
                    # Calculate base energy regardless of cutoff
                    d_norm = distance / cutoff

                    if distance < 4.5:  # Very close in space
                        base_energy = contact_energy * (1.0 - d_norm**2)
                    elif distance < 6.0:  # Medium range
                        base_energy = contact_energy * 0.5 * (1.0 - d_norm)
                    elif distance < cutoff:  # Longer range
                        base_energy = contact_energy * 0.2 * (1.0 - d_norm)
                    else:
                        base_energy = 0.0  # No contact energy beyond cutoff

                    # Adjust the scaling factors for sequence separation
                    if seq_sep <= 1:
                        energy = (
                            base_energy * 0.1
                        )  # Slight reduction for adjacent residues
                    else:
                        energy = base_energy
                        if seq_sep > 10:
                            energy *= 1.2  # Boost very long-range contacts

                    # Store contact energies symmetrically
                    contact_map[i - 1, j - 1] = energy
                    contact_map[j - 1, i - 1] = energy
            else:
                # Handle missing atoms
                distance_map[i - 1, j - 1] = max_distance
                distance_map[j - 1, i - 1] = max_distance

    # Set diagonal to zero
    np.fill_diagonal(distance_map, 0.0)
    np.fill_diagonal(contact_map, 0.0)

    # Save maps
    np.savetxt(output_dm, distance_map, fmt="%.6e")
    np.savetxt(output_cm, contact_map, fmt="%.6e")

    return distance_map, contact_map


def verify_maps(distance_map, contact_map, reference_cm=None):
    """
    Verify the generated maps match expected properties and compare with reference
    """
    logger.info("\nVerifying maps:")
    logger.info("-" * 50)

    # Basic checks
    assert distance_map.shape == (60, 60), "Distance map should be 60x60"
    assert contact_map.shape == (60, 60), "Contact map should be 60x60"
    logger.info("✓ Dimensions check passed (60x60)")

    assert np.allclose(distance_map, distance_map.T), "Distance map should be symmetric"
    assert np.allclose(contact_map, contact_map.T), "Contact map should be symmetric"
    logger.info("✓ Symmetry check passed")

    assert np.all(np.diag(distance_map) == 0), "Distance map diagonal should be zero"
    assert np.all(np.diag(contact_map) == 0), "Contact map diagonal should be zero"
    logger.info("✓ Diagonal check passed")

    assert np.all(distance_map >= 0), "Distances should be non-negative"
    # Contact energies may be negative or zero
    logger.info("✓ Value range check passed")

    # Compare with reference if provided
    if reference_cm is not None:
        logger.info("\nComparing with reference contact map:")
        max_diff = np.max(np.abs(contact_map - reference_cm))
        logger.info(f"Maximum absolute difference: {max_diff:.6e}")

        # Log detailed differences
        threshold = 1e-6
        differences = np.where(np.abs(contact_map - reference_cm) > threshold)
        if len(differences[0]) > 0:
            logger.info("\nSignificant differences found:")
            for i, j in zip(*differences):
                logger.info(
                    f"Position ({i+1},{j+1}): Generated={contact_map[i,j]:.6e}, Reference={reference_cm[i,j]:.6e}"
                )
        else:
            logger.info("✓ Maps match within tolerance")


def main():
    import os
    from urllib.request import urlretrieve

    # Load reference contact map
    reference_cm = load_reference_map("data/srcsh3_cm.dat")
    if reference_cm is None:
        logger.warning("Could not load reference contact map")

    pdb_id = "4JZ4"
    pdb_file = f"{pdb_id}.pdb"

    if not os.path.exists(pdb_file):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urlretrieve(url, pdb_file)
        logger.info(f"Downloaded {pdb_file}")

    # Generate maps
    distance_map, contact_map = generate_distance_contact_maps(
        pdb_file=pdb_file,
        output_dm="srcsh3_dm.dat",
        output_cm="srcsh3_cm.dat",
        n_residues=60,
        cutoff=8.0,
        contact_energy=-2.506,  # From Table 2
        chain_id="A",
    )

    # Verify maps and compare with reference
    verify_maps(distance_map, contact_map, reference_cm)


if __name__ == "__main__":
    main()
