import sys
import re
import io
import contextlib
from typing import List, Dict

#这个程序旨在将汇编语言转换为机器码，支持基本指令和伪指令，并可以处理标签和映射。
#仅支持一对.ORIG和.END指令，输入通过标准输入提供，输出为机器码的二进制表示。
#.END指令后忽略所有内容，注释以;开头
#输出格式内含起始地址的16位二进制表示，后跟每条指令的16位二进制表示


#指令集，没有1101
OPCODES = {
    'BR': '0000', 'ADD': '0001', 'LD': '0010', 'ST': '0011', 'JSR': '0100', 'JSRR': '0100',
    'AND': '0101', 'LDR': '0110', 'STR': '0111', 'RTI': '1000', 'NOT': '1001', 'LDI': '1010',
    'STI': '1011', 'JMP': '1100', 'LEA': '1110', 'TRAP': '1111', 'RET': '1101'
}

BRFamily = {'BR', 'BRN', 'BRZ', 'BRP', 'BRNZ', 'BRNP', 'BRZP', 'BRNZP'}
#TRAP代码映射
TRAPS = {
    'GETC': 0x20, 'OUT': 0x21, 'PUTS': 0x22, 'IN': 0x23, 'PUTSP': 0x24, 'HALT': 0x25
}

#伪指令集合
PSEUDOS = {'.FILL', '.BLKW', '.STRINGZ'}

def to_twos_complement(value: int, bits: int) -> int:
    #将整数转换为指定位数的二进制补码表示
    mask = (1 << bits) - 1
    return value & mask

def parse_num(token: str) -> int:
    # 十进制和十六进制的数字解析，返回整数值
    token = token.strip()
    if token.startswith('#'):
        return int(token[1:])
    if token.startswith('x') or token.startswith('X'):
        return int(token[1:], 16)
    return int(token)

def reg_num(token: str) -> int:
    # 返回寄存器编号
    token = token.strip()
    if token.upper().startswith('R'):
        return int(token[1:])
    raise ValueError(f"Invalid register {token}")


def assembler_core() -> None:
    """原始命令行版本的主流程：从标准输入读取，向标准输出打印结果。"""
    # 读取起始地址.ORIG
    ORIG = "NULL"
    while not ORIG.upper().startswith('.ORIG'):
        ORIG = input()
    START_ADDR = int(ORIG.split()[1][1:], 16)
    PC = START_ADDR

    # 按行读取文件直到.END
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line is None:
            break
        if line.strip().upper().startswith('.END'):
            break
        if ';' in line:
            line = line.split(';', 1)[0]
        if not line.strip():
            continue
        lines.append(line.rstrip())

    # 第一轮扫描：记录标签地址并分割行元素
    label_addr: Dict[str, int] = {}
    addr = START_ADDR
    parsed_lines = []  # (line, label, opcode, operands_str, addr)

    for raw in lines:
        s = raw.strip()
        tokens = s.split()
        label = None
        opcode = None
        operands_str = ''
        if tokens:
            first = tokens[0]
            # 判断第一个token是标签还是指令
            if first.upper() in OPCODES or first.upper() in TRAPS or first.upper() in PSEUDOS or first.upper() in BRFamily:
                opcode = first
                operands_str = s[len(opcode):].strip()
            else:
                # 标签正则化
                label = first.rstrip(':')   #去掉冒号
                norm_label = label.upper()  #大写
                if norm_label in label_addr:
                    raise ValueError(f"Duplicate label {label}")    #重复标签报错
                label_addr[norm_label] = addr                       #加入标签表
                rest = s[s.find(tokens[1]):].strip() if len(tokens) > 1 else ''  #处理标签后空行
                if rest:
                    opcode = rest.split()[0]
                    operands_str = rest[rest.upper().find(opcode.upper()) + len(opcode):].strip()
        parsed_lines.append((label, opcode, operands_str, addr)) #生成分割行列表
        if opcode is None:
            continue    # 空行或仅标签行
        op = opcode.upper()
        if op in OPCODES or op.startswith('BR') or op in TRAPS:
            addr += 1
        elif op == '.FILL':
            addr += 1
        elif op == '.BLKW':
            block_size = parse_num(operands_str.split()[0]) if operands_str else 0
            addr += block_size
        elif op == '.STRINGZ':
            m = re.search(r'"(.*)"', operands_str)
            string_val = m.group(1) if m else ''
            addr += len(string_val) + 1

    # 第二轮扫描：生成机器码
    output_binaries: List[str] = []
    addr = START_ADDR
    #遍历分割行列表并正则化处理
    for label, opcode, operands_str, line_addr in parsed_lines:
        if opcode is None:
            continue
        op = opcode.upper()
        operands = [o.strip() for o in operands_str.split(',') if o.strip()]

        #具体操作处理

        if op == '.FILL':
            token = operands[0]
            try:
                val = parse_num(token)
            except Exception:
                # allow a label as the value for .FILL
                key = token.rstrip(':').upper()
                if key not in label_addr:
                    raise ValueError(f"Undefined label {token}")
                val = label_addr[key]
            output_binaries.append(format(to_twos_complement(val, 16), '016b'))
            addr += 1
            continue
        if op == '.BLKW':
            block_size = parse_num(operands[0])
            for _ in range(block_size):
                output_binaries.append('0' * 16)
            addr += block_size
            continue
        if op == '.STRINGZ':
            m = re.search(r'"(.*)"', operands_str)
            string_val = m.group(1) if m else ''
            for ch in string_val:
                output_binaries.append(format(ord(ch) & 0xFFFF, '016b'))
            output_binaries.append('0' * 16)
            addr += len(string_val) + 1
            continue

        # BR系列指令处理
        if op in BRFamily:
            flags = 0
            suffix = op[2:]
            if suffix == '':
                flags = 0b111
            else:                   #判断CC位
                if 'N' in suffix:
                    flags |= 0b100
                if 'Z' in suffix:
                    flags |= 0b010
                if 'P' in suffix:
                    flags |= 0b001 
            dest = operands[0].rstrip(':').upper()
            if dest not in label_addr:
                raise ValueError(f"Undefined label {operands[0]}")
            pc_offset = label_addr[dest] - (line_addr + 1)
            if pc_offset < -256 or pc_offset > 255:
                raise ValueError(f"Offset out of range for BR to {dest}")
            imm9 = to_twos_complement(pc_offset, 9)
            binary = OPCODES['BR'] + format(flags, '03b') + format(imm9, '09b')
            output_binaries.append(binary)
            addr += 1
            continue
        #ADD和AND指令处理，由于格式相似合并处理
        if op == 'ADD' or op == 'AND':
            opcode_bits = OPCODES[op]
            DR = reg_num(operands[0])
            SR1 = reg_num(operands[1])
            third = operands[2]
            if third.upper().startswith('R'):
                SR2 = reg_num(third)
                binary = opcode_bits + format(DR, '03b') + format(SR1, '03b') + '0' + format(0, '02b') + format(SR2, '03b')
            else:
                imm5 = parse_num(third)
                binary = opcode_bits + format(DR, '03b') + format(SR1, '03b') + '1' + format(to_twos_complement(imm5, 5), '05b')
            output_binaries.append(binary)
            addr += 1
            continue
        if op == 'NOT':
            opcode_bits = OPCODES['NOT']
            DR = reg_num(operands[0])
            SR = reg_num(operands[1])
            binary = opcode_bits + format(DR, '03b') + format(SR, '03b') + '1' * 6
            output_binaries.append(binary)
            addr += 1
            continue
        if op in {'LD', 'ST', 'LEA', 'LDI', 'STI'}:
            opcode_bits = OPCODES['LD'] if op == 'LD' else OPCODES[op]
            R = reg_num(operands[0])
            #判断是否为立即数
            if operands[1].upper().startswith('#'):
                imm9 = parse_num(operands[1][1:])
                binary = opcode_bits + format(R, '03b') + format(imm9, '09b')
                output_binaries.append(binary)
                addr += 1
                continue
            elif operands[1].upper().startswith('X') and operands[1][1:].isdigit():
                imm9 = parse_num(operands[1][1:])
                binary = opcode_bits + format(R, '03b') + format(to_twos_complement(imm9, 9), '09b')
                output_binaries.append(binary)
                addr += 1
                continue
            #标签处理
            dest = operands[1].rstrip(':').upper()
            if dest not in label_addr:
                raise ValueError(f"Undefined label {operands[1]}")
            pc_offset = label_addr[dest] - (line_addr + 1)
            if pc_offset < -(1 << 8) or pc_offset > (1 << 8) - 1:
                raise ValueError(f"Offset out of range for {op} to {dest}")
            imm9 = to_twos_complement(pc_offset, 9)
            binary = opcode_bits + format(R, '03b') + format(imm9, '09b')
            output_binaries.append(binary)
            addr += 1
            continue
        if op == 'LDR' or op == 'STR':
            opcode_bits = OPCODES[op]
            R1 = reg_num(operands[0])
            BaseR = reg_num(operands[1])
            offset6 = parse_num(operands[2])
            binary = opcode_bits + format(R1, '03b') + format(BaseR, '03b') + format(to_twos_complement(offset6, 6), '06b')
            output_binaries.append(binary)
            addr += 1
            continue
        if op == 'RTI':
            binary = OPCODES['RTI'] + '000000000000'
            output_binaries.append(binary)
            addr += 1
            continue
        if op == 'JMP' or op == 'RET':
            opcode_bits = OPCODES['JMP']
            if op == 'RET':
                binary = opcode_bits + '000' + format(7, '03b') + '000000'
            else:
                BaseR = reg_num(operands[0])
                binary = opcode_bits + '000' + format(BaseR, '03b') + '000000'
            output_binaries.append(binary)
            addr += 1
            continue
        if op == 'JSR' or op == 'JSRR':
            opcode_bits = OPCODES['JSR']
            if op == 'JSR' and not operands[0].upper() in {'R0','R1','R2','R3','R4','R5','R6','R7'}:
                dest = operands[0].rstrip(':').upper()
                if dest not in label_addr:
                    raise ValueError(f"Undefined label {operands[0]}")
                pc_offset = label_addr[dest] - (line_addr + 1)
                if pc_offset < -(1 << 10) or pc_offset > (1 << 10) - 1:
                    raise ValueError("Offset out of range for JSR")
                imm11 = to_twos_complement(pc_offset, 11)
                binary = opcode_bits + '1' + format(imm11, '011b')
            else:
                BaseR = reg_num(operands[0])
                binary = opcode_bits + '0' + '00' + format(BaseR, '03b') + '000000'
            output_binaries.append(binary)
            addr += 1
            continue

        if op == 'TRAP' or op in TRAPS:
            if op == 'TRAP':
                trap_code = parse_num(operands[0])
            else:
                trap_code = TRAPS[op]
            binary = OPCODES['TRAP'] + '0000' + format(trap_code & 0xFF, '08b')
            output_binaries.append(binary)
            addr += 1
            continue
        if op in TRAPS:
            trap_code = TRAPS[op]
            binary = OPCODES['TRAP'] + '0000' + format(trap_code & 0xFF, '08b')
            output_binaries.append(binary)
            addr += 1
            continue
        raise ValueError(f"Unhandled opcode: {op} at address: {format(to_twos_complement(line_addr,16), '04X')}")

    # 输出起始地址和机器码
    print(format(START_ADDR, '016b'))
    for b in output_binaries:
        print(b)


def assemble_code(asm_source: str) -> str:
    """
    提供给可视化界面的装配函数：
    - 输入：完整的汇编代码字符串（包含 .ORIG / .END 等）
    - 输出：起始地址 + 每条指令机器码，每行一条的字符串
    """
    buffer_in = io.StringIO(asm_source)
    buffer_out = io.StringIO()

    old_stdin = sys.stdin
    try:
        sys.stdin = buffer_in
        with contextlib.redirect_stdout(buffer_out):
            assembler_core()
    finally:
        sys.stdin = old_stdin

    return buffer_out.getvalue()


def assemble_code_lines(asm_source: str) -> List[str]:
    """
    调用装配器并返回每一行机器码（包括起始地址），方便对比。
    会自动去除空行和首尾空白。
    """
    text = assemble_code(asm_source)
    return [line.strip() for line in text.splitlines() if line.strip()]


def check_code(output_binaries: List[str], answer_binaries: List[str]) -> bool:
    if len(output_binaries)!=len(answer_binaries):
        return False
    for i in range(len(output_binaries)):
        if output_binaries[i]!=answer_binaries[i]:
            return False
    return True
    
def main():
    """基于 Streamlit 的简单可视化界面。"""
    import streamlit as st

    st.set_page_config(page_title="LC-3 汇编器可视化", layout="wide")
    st.title("LC-3 汇编器可视化")
    st.markdown(
        """
**说明：**
- 在左侧文本框中输入完整的 LC-3 汇编代码（包含 `.ORIG` 和 `.END`）。
- 点击“转换为机器码”按钮后，右侧会显示输出的二进制机器码。
- 转换逻辑完全复用原有命令行程序，只是将输出内容集中展示在页面中。
"""
    )

    col1, col2 = st.columns(2)

    with col1:
        default_asm = """.ORIG x3000
ADD R1, R1, #1
BRz DONE
DONE HALT
.END
"""
        asm_text = st.text_area(
            "在此粘贴或编写 LC-3 汇编代码：",
            value=default_asm,
            height=260,
            key="asm_input",
        )

        if st.button("转换为机器码"):
            if not asm_text.strip():
                st.warning("请输入汇编码再进行转换。")
            else:
                try:
                    compiled_text = assemble_code(asm_text)
                    compiled_lines = assemble_code_lines(asm_text)
                    st.session_state["compiled_text"] = compiled_text
                    st.session_state["compiled_lines"] = compiled_lines
                except Exception as e:
                    st.error(f"汇编过程中发生错误：{e}")

        st.markdown("---")
        st.subheader("答案核对")
        st.markdown(
            "在下面输入你认为**正确的机器码**，一行一条（包括起始地址那一行），"
            "点击“核对答案”进行比较。"
        )
        answer_text = st.text_area(
            "在此粘贴/输入你的参考答案机器码：",
            height=200,
            key="answer_input",
        )
        if st.button("核对答案"):
            if "compiled_lines" not in st.session_state:
                st.warning("请先点击“转换为机器码”生成一份结果，再进行核对。")
            else:
                compiled_lines = st.session_state.get("compiled_lines", [])
                answer_lines = [
                    line.strip()
                    for line in answer_text.splitlines()
                    if line.strip()
                ]
                if not answer_lines:
                    st.warning("请先在上方输入参考答案的机器码。")
                else:
                    ok = check_code(compiled_lines, answer_lines)
                    if ok:
                        st.success("✅ 编译结果与参考答案 **完全一致**。")
                    else:
                        # 找出第一个不一致的位置
                        min_len = min(len(compiled_lines), len(answer_lines))
                        mismatch_index = None
                        for i in range(min_len):
                            if compiled_lines[i] != answer_lines[i]:
                                mismatch_index = i
                                break
                        if mismatch_index is None and len(compiled_lines) != len(answer_lines):
                            mismatch_index = min_len

                        st.error("❌ 编译结果与参考答案 **不一致**。")
                        with st.expander("查看详细对比"):
                            st.write(f"编译结果行数：{len(compiled_lines)}")
                            st.write(f"答案行数：{len(answer_lines)}")
                            if mismatch_index is not None:
                                st.write(f"首个不一致出现在第 {mismatch_index} 行（从 0 开始计数）。")
                                got = compiled_lines[mismatch_index] if mismatch_index < len(compiled_lines) else "<无>"
                                expect = answer_lines[mismatch_index] if mismatch_index < len(answer_lines) else "<无>"
                                st.code(
                                    f"编译结果: {got}\n参考答案: {expect}",
                                    language="text",
                                )

    with col2:
        st.subheader("机器码输出")
        compiled_text = st.session_state.get("compiled_text", "")
        if compiled_text:
            st.code(compiled_text, language="text")
        else:
            st.info("暂未生成机器码，请在左侧输入汇编并点击“转换为机器码”。")


if __name__ == "__main__":
    main()
