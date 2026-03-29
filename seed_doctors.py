"""
Seed script for GAIA dummy doctors across Lebanon.
Adds specialty column if missing, then inserts doctors.
Coordinates are stored as "lat, lng" in the location field.
Usage: python seed_doctors.py
"""

from app.db import SessionLocal, engine
from app.models import User
from app.auth import hash_password
from sqlalchemy import text


def run_migrations():
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS specialty TEXT"))
    print("Migration OK: specialty column ensured.")


# Format: (name, email, specialty, phone, lat, lng)
DOCTORS = [
    # Allergist / Immunologist
    ("Dr. Layla Khoury",       "layla.khoury@gaia.lb",       "Allergist / Immunologist",                          "+961 1 234001", 33.8938, 35.4925),
    ("Dr. Samir Gemayel",      "samir.gemayel@gaia.lb",      "Allergist / Immunologist",                          "+961 9 234002", 34.0072, 35.6481),

    # Cardiologist
    ("Dr. Nadia Haddad",       "nadia.haddad@gaia.lb",       "Cardiologist",                                      "+961 1 234003", 33.8872, 35.5131),
    ("Dr. Fadi Nassar",        "fadi.nassar@gaia.lb",        "Cardiologist",                                      "+961 6 234004", 34.4367, 35.8201),

    # Dentist / Periodontist
    ("Dr. Rima Saleh",         "rima.saleh@gaia.lb",         "Dentist / Periodontist",                            "+961 1 234005", 33.8731, 35.4892),
    ("Dr. Elie Azar",          "elie.azar@gaia.lb",          "Dentist / Periodontist",                            "+961 4 234006", 34.1236, 35.6517),

    # Dermatologist
    ("Dr. Maya Frem",          "maya.frem@gaia.lb",          "Dermatologist",                                     "+961 1 234007", 33.8893, 35.4979),
    ("Dr. Chadi Rizk",         "chadi.rizk@gaia.lb",         "Dermatologist",                                     "+961 8 234008", 33.8469, 35.9019),

    # Endocrinologist
    ("Dr. Hala Moussa",        "hala.moussa@gaia.lb",        "Endocrinologist",                                   "+961 1 234009", 33.8916, 35.4881),
    ("Dr. Georges Khalil",     "georges.khalil@gaia.lb",     "Endocrinologist",                                   "+961 7 234010", 33.5472, 35.3808),

    # Gastroenterologist / Hepatologist / Proctologist
    ("Dr. Maroun Tawk",        "maroun.tawk@gaia.lb",        "Gastroenterologist / Hepatologist / Proctologist",  "+961 1 234011", 33.8839, 35.5176),
    ("Dr. Joelle Daoud",       "joelle.daoud@gaia.lb",       "Gastroenterologist / Hepatologist / Proctologist",  "+961 6 234012", 34.4522, 35.8453),

    # Gynecologist / OB-GYN
    ("Dr. Sandra Wehbe",       "sandra.wehbe@gaia.lb",       "Gynecologist / Obstetrician (OB/GYN)",              "+961 1 234013", 33.8700, 35.4937),
    ("Dr. Roula Abboud",       "roula.abboud@gaia.lb",       "Gynecologist / Obstetrician (OB/GYN)",              "+961 9 234014", 34.2553, 35.6586),

    # Hematologist
    ("Dr. Karim Saad",         "karim.saad@gaia.lb",         "Hematologist",                                      "+961 1 234015", 33.8956, 35.4841),
    ("Dr. Lara Moukalled",     "lara.moukalled@gaia.lb",     "Hematologist",                                      "+961 8 234016", 34.0042, 36.2118),

    # Nephrologist
    ("Dr. Tony Geagea",        "tony.geagea@gaia.lb",        "Nephrologist",                                      "+961 1 234017", 33.9003, 35.4732),
    ("Dr. Aline Karam",        "aline.karam@gaia.lb",        "Nephrologist",                                      "+961 5 234018", 33.8083, 35.5997),

    # Neurologist
    ("Dr. Ziad Mansour",       "ziad.mansour@gaia.lb",       "Neurologist",                                       "+961 1 234019", 33.8682, 35.5043),
    ("Dr. Nadine Jreissati",   "nadine.jreissati@gaia.lb",   "Neurologist",                                       "+961 9 234020", 33.9941, 35.6531),

    # Ophthalmologist
    ("Dr. Pierre Feghali",     "pierre.feghali@gaia.lb",     "Ophthalmologist",                                   "+961 1 234021", 33.8938, 35.5018),
    ("Dr. Carmen Sleiman",     "carmen.sleiman@gaia.lb",     "Ophthalmologist",                                   "+961 6 234022", 34.4431, 35.8378),

    # Orthopedic Surgeon / Rheumatologist
    ("Dr. Antoine Khoueiry",   "antoine.khoueiry@gaia.lb",   "Orthopedic Surgeon / Rheumatologist",               "+961 1 234023", 33.8762, 35.5219),
    ("Dr. Mireille Hajj",      "mireille.hajj@gaia.lb",      "Orthopedic Surgeon / Rheumatologist",               "+961 7 234024", 33.5630, 35.3731),

    # Otolaryngologist (ENT)
    ("Dr. Ramzi Bou Khalil",   "ramzi.boukhalil@gaia.lb",    "Otolaryngologist (ENT)",                            "+961 1 234025", 33.8978, 35.5127),
    ("Dr. Celine Nasr",        "celine.nasr@gaia.lb",        "Otolaryngologist (ENT)",                            "+961 4 234026", 34.1183, 35.6494),

    # Pediatrician
    ("Dr. Dany Chaoul",        "dany.chaoul@gaia.lb",        "Pediatrician",                                      "+961 1 234027", 33.8845, 35.5164),
    ("Dr. Tania Sassine",      "tania.sassine@gaia.lb",      "Pediatrician",                                      "+961 8 234028", 33.8700, 35.9200),

    # Psychiatrist
    ("Dr. Ghassan Rahhal",     "ghassan.rahhal@gaia.lb",     "Psychiatrist",                                      "+961 1 234029", 33.8748, 35.4869),
    ("Dr. Joanna Makhoul",     "joanna.makhoul@gaia.lb",     "Psychiatrist",                                      "+961 6 234030", 34.4612, 35.8297),

    # Pulmonologist / Sleep Specialist
    ("Dr. Elias Bitar",        "elias.bitar@gaia.lb",        "Pulmonologist / Sleep Specialist",                  "+961 1 234031", 33.8981, 35.4857),
    ("Dr. Pamela Assaf",       "pamela.assaf@gaia.lb",       "Pulmonologist / Sleep Specialist",                  "+961 7 234032", 33.3772, 35.4836),

    # Urologist
    ("Dr. Joseph Abou Jaoude", "joseph.aboujaoude@gaia.lb",  "Urologist",                                         "+961 1 234033", 33.8858, 35.5498),
    ("Dr. Mirna Ghosn",        "mirna.ghosn@gaia.lb",        "Urologist",                                         "+961 3 234034", 33.2705, 35.2038),

    # General Surgery / Critical Care / Pain Management
    ("Dr. Habib Frem",         "habib.frem@gaia.lb",         "General Surgery / Critical Care / Pain Management", "+961 1 234035", 33.8828, 35.5058),
    ("Dr. Nathalie Bou Habib", "nathalie.bouhabib@gaia.lb",  "General Surgery / Critical Care / Pain Management", "+961 5 234036", 33.6914, 35.5794),
]


def seed_doctors():
    db = SessionLocal()
    created = 0
    skipped = 0
    try:
        for name, email, specialty, phone, lat, lng in DOCTORS:
            if db.query(User).filter(User.email == email).first():
                skipped += 1
                continue
            doctor = User(
                name=name,
                email=email,
                password_hash=hash_password("Doctor@12345"),
                role="doctor",
                is_active=True,
                specialty=specialty,
                phone=phone,
                location=f"{lat}, {lng}",
            )
            db.add(doctor)
            created += 1
        db.commit()
        print(f"Done. Created: {created}  Skipped (already exist): {skipped}")
    except Exception as e:
        db.rollback()
        print(f"Error: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("Running migrations...")
    run_migrations()
    print("Seeding doctors...")
    seed_doctors()
