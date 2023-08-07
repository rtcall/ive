ENTRY(_start);

SECTIONS
{
	. = 0x80000000;
	.text : { *(.text .text.*) }
	. = 0x0;
	.data : { *(.data) }
	.bss : { *(.bss) }
}
