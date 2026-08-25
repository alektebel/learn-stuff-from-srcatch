# Linux From Scratch

Build an operating system from the first byte the CPU executes to a self-hosting
Unix-like system — and, in parallel, build a real Linux distribution from source
so that nothing running on your machine is there without you having put it there.

**This directory is scaffolding only.** It contains no solutions and no exercise
code: it is the *frame* — the division of the work into 147 exercises across 13
phases, the limit case that motivates each one, and the book or paper you should
be reading while you do it. You write the code. That is the entire point.

---

## The honest scope

"Linux from scratch" means two genuinely different projects, and conflating them
is the most common way people waste a year:

| | **Track K — your own kernel** | **Track D — your own distribution** |
|---|---|---|
| What you build | An OS kernel, from a boot sector to processes, filesystems and TCP | A working Linux system, from a cross-toolchain to a booting userland |
| What you learn | *Why* an OS is shaped the way it is — you invent each mechanism after hitting the failure that requires it | *What is actually on a Linux box* — every binary, why it exists, who put it there |
| The source | OSDev tradition, xv6, MINIX, the Intel SDM | The [Linux From Scratch](https://www.linuxfromscratch.org/) book, Beyond LFS |
| Phases | 00 – 10, 12 | 11 (and it is large) |
| Time, realistically | 6 – 18 months, part time | 2 – 6 weekends for a first pass |

You asked for both, and both are here. **Do Track D once, early, in parallel with
phase 02** — it is short, and it gives you a real system to compare yours against
for the rest of the project. Then spend the year on Track K.

They meet in phase 12, where your kernel boots the userland you built in phase 11.

---

## The ladder

Every phase exists because the previous one broke. This is principle 2 of
[PHILOSOPHY.md](../PHILOSOPHY.md) applied to an entire OS:

```
 00  a cross-compiler that targets nothing              your host libc is a lie
 01  16 bytes that print a character                    ...512 bytes is not enough room
 02  a kernel with interrupts and a console             ...a fault has nowhere to go
 03  paging and a heap                                  ...two programs see one address space
 04  processes, syscalls, a scheduler                   ...one of them never yields
 05  locks and a second CPU                             ...two cores enter the same critical section
 06  a block device, a VFS, a filesystem                ...power fails mid-write
 07  ELF loading, libc, fork/exec, a shell              ...it can only talk to the console
 08  PCI, DMA, a driver model                           ...every driver reimplements the same bug
 09  a NIC driver and a TCP stack                       ...an ack arrives for data you already freed
 10  users, namespaces, seccomp                         ...any process can read any file
 11  a real toolchain, a real kernel, a real userland   ...it is 400 packages and you trust none of them
 12  your kernel runs that userland                     ...you are now maintaining an OS
```

Read that column of failures. Each one is an exercise you are told to *construct
deliberately*, not avoid. An OS you have never crashed is an OS you do not
understand.

---

## Structure

```
linux-from-scratch/
├── README.md            # this file — the frame and the two tracks
├── REFERENCES.md        # the bibliography every phase cites by [TAG]
├── PROGRESS.md          # all 147 exercises as checkboxes
├── phase-00-toolchain/          README.md   —  6 exercises
├── phase-01-boot/               README.md   — 10 exercises
├── phase-02-kernel-core/        README.md   — 12 exercises
├── phase-03-memory/             README.md   — 13 exercises
├── phase-04-processes/          README.md   — 13 exercises
├── phase-05-concurrency-smp/    README.md   — 12 exercises
├── phase-06-storage-fs/         README.md   — 14 exercises
├── phase-07-userspace/          README.md   — 14 exercises
├── phase-08-drivers/            README.md   — 11 exercises
├── phase-09-networking/         README.md   — 14 exercises
├── phase-10-security-isolation/ README.md   — 12 exercises
├── phase-11-real-linux-lfs/     README.md   — 10 exercises
└── phase-12-capstone/           README.md   —  6 exercises
```

Each phase README has the same shape:

- **The failure that got you here** — what broke in the previous phase.
- **Design decisions** — the forks in the road, stated as decisions with costs,
  because a finished kernel hides its own reasoning.
- **The exercises** — numbered `NN.M`, each with *what you build*, *the limit
  case*, *done when*, and *read*.
- **Where this phase stops** — what real Linux does that you are deliberately
  not doing, so you know the boundary you are walking past.

Every `read` entry cites [REFERENCES.md](REFERENCES.md) by tag, e.g. `[OSTEP §13-16]`,
`[P-FFS84]`, `[SDM3 ch.5]`.

---

## Ground rules

**Everything runs in an emulator first.** QEMU with `-s -S` and GDB attached, so
a triple fault is a debugging session and not a reboot. Real hardware is phase 12.

**Write it in C and assembly.** Not because C is good, but because every document
you will read for the next year — the SDM, the ELF ABI, `Documentation/` in the
kernel tree, every OSDev page — assumes it. Fighting your toolchain and the
hardware at once is one fight too many. (Rust is a legitimate second pass; do it
after you have done it once in C, when the borrow checker is arguing with you
about something you already understand.)

**Commit after every exercise, and write in the message what broke.** The log of
how your kernel failed is the most valuable artifact this project produces. You
will not remember why you added that `wbinvd` in six weeks.

**Never fix a bug you cannot explain.** A kernel bug that goes away when you add
a print statement has not gone away.

**Measure, don't assume.** Every phase ends with numbers you print yourself —
context switches per second, TLB miss cost, syscall latency, packets per second.
Predict each one before you look at it. A surprise is a gap in your model.

---

## Verification

The rest of this repo ships `check.py` per directory. This one cannot, yet: there
is no code to check until you write it. The equivalent discipline here is that
**every exercise states a `done when` that is an observable event** — a register
value, a QEMU log line, a `readelf` output, a packet on the wire — never "it
seems to work".

As you complete a phase, write its checker. `PROGRESS.md` tracks both.

---

## Prerequisites, stated plainly

You need C (pointers, `volatile`, `union`, the linker's view of a program),
comfort at a shell, and enough x86-64 assembly to read a disassembly. You do not
need prior OS knowledge — phase 00 assumes none.

If C is shaky, [c-compiler/](../c-compiler/) and [bash-from-scratch/](../bash-from-scratch/)
in this repo are the warm-up. If the *machine* is shaky — cache, virtual memory,
linking as concepts — read `[CSAPP]` chapters 3, 6, 7 and 9 first. Three weeks
there saves three months here.

---

## Where the whole project stops

Deliberately out of scope, so you know the edge:

- **Multiple architectures.** x86-64 only, with ARM64 noted where the difference
  teaches something. Porting is a fine second year.
- **Formal verification.** seL4 proves its kernel correct `[P-SEL4]`; you are not
  going to, and the exercises do not pretend otherwise.
- **A GUI.** Framebuffer and a windowing protocol are sketched in phase 08 and
  left there. Wayland is another project.
- **Being fast.** Correct first, measured second, optimised only where the
  measurement demanded it. Your scheduler will lose to CFS. That is fine — the
  exercise is to find out *where* it loses, and why.
- **Being safe to run on hardware you care about.** Phase 12 uses a spare machine
  or a USB stick, never your laptop's only disk.
