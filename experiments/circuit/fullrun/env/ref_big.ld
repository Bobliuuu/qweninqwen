/* Enlarged-memory map for the full-model reference run (software model only): code at the real ROM base 0x40000 (so AUIPC/JAL link values equal the real map),
   everything else (rodata, data, bss, stack) in a 4 MB RAM relocated to 0x1000000 (+0x100). The weight blob is the DROM at 0x10000000 (loaded by the driver, not linked). */
ENTRY(_start)
SECTIONS
{
  . = 0x40000;
  .text : { *(.text.init) *(.text .text.*) }
  . = 0x1000100;
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) *(.init_array .init_array.* .fini_array .fini_array.* .eh_frame .eh_frame_hdr) }
  .data : { *(.data .data.* .sdata .sdata.*) }
  .tdata : { __tls_base = .; *(.tdata .tdata.*) }
  .tbss : { *(.tbss .tbss.*) }
  .bss : {
    . = . + SIZEOF(.tbss);
    *(.bss .bss.* .sbss .sbss.*) *(COMMON)
    . = ALIGN(16);
    __heap_start = .;
  }
  _end = .;
}
