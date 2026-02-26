from dataclasses import dataclass, field
from typing import List

import streamlit as st


TRAPS = {
    "GETC": 0x20,
    "OUT": 0x21,
    "PUTS": 0x22,
    "IN": 0x23,
    "PUTSP": 0x24,
    "HALT": 0x25,
}


class Memory:
    def __init__(self) -> None:
        # 16 位地址空间
        self.code: List[str] = ["0000000000000000"] * 65536


@dataclass
class LC3State:
    memory: Memory
    registers: List[int] = field(default_factory=lambda: [0] * 8)  # R0-R7
    pc: int = 0
    cc: str = "Z"  # N / Z / P
    halted: bool = False
    last_executed_pc: int | None = None  # 刚刚执行的指令地址
    last_change: dict | None = None      # 上一步状态变化：{"type": "reg"/"mem"/"cc"/"pc", ...}


def sign_extend(value: int, bits: int) -> int:
    """按 LC-3 规则对补码立即数进行符号扩展。"""
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def disassemble(instruction: str, pc: int) -> str:
    """
    将 16 位二进制机器码反汇编为一条近似的 LC-3 汇编码（不包含标签）。
    pc 为该指令当前的地址（即将执行时的 PC）。
    """
    if len(instruction) != 16 or any(c not in "01" for c in instruction):
        return "<invalid>"

    op = instruction[0:4]

    def r(n: int) -> str:
        return f"R{n}"

    # ADD / AND
    if op == "0001" or op == "0101":
        mnem = "ADD" if op == "0001" else "AND"
        DR = int(instruction[4:7], 2)
        SR1 = int(instruction[7:10], 2)
        if instruction[10] == "0":
            SR2 = int(instruction[13:16], 2)
            return f"{mnem} {r(DR)}, {r(SR1)}, {r(SR2)}"
        else:
            imm5 = sign_extend(int(instruction[11:16], 2), 5)
            return f"{mnem} {r(DR)}, {r(SR1)}, #{imm5}"

    # BR / BRnzp 等
    if op == "0000":
        n = instruction[4] == "1"
        z = instruction[5] == "1"
        p = instruction[6] == "1"
        cond = ""
        cond += "N" if n else ""
        cond += "Z" if z else ""
        cond += "P" if p else ""
        mnem = "BR" + cond if cond else "BR"
        off9 = sign_extend(int(instruction[7:16], 2), 9)
        target = (pc + 1 + off9) & 0xFFFF
        return f"{mnem} x{target:04X}"

    # LD / ST / LEA / LDI / STI（PC 相对）
    if op in {"0010", "0011", "1110", "1010", "1011"}:
        DR_SR = int(instruction[4:7], 2)
        off9 = sign_extend(int(instruction[7:16], 2), 9)
        target = (pc + 1 + off9) & 0xFFFF
        if op == "0010":
            return f"LD {r(DR_SR)}, x{target:04X}"
        if op == "0011":
            return f"ST {r(DR_SR)}, x{target:04X}"
        if op == "1110":
            return f"LEA {r(DR_SR)}, x{target:04X}"
        if op == "1010":
            return f"LDI {r(DR_SR)}, x{target:04X}"
        if op == "1011":
            return f"STI {r(DR_SR)}, x{target:04X}"

    # LDR / STR
    if op == "0110" or op == "0111":
        mnem = "LDR" if op == "0110" else "STR"
        DR_SR = int(instruction[4:7], 2)
        BaseR = int(instruction[7:10], 2)
        off6 = sign_extend(int(instruction[10:16], 2), 6)
        return f"{mnem} {r(DR_SR)}, {r(BaseR)}, #{off6}"

    # NOT
    if op == "1001":
        DR = int(instruction[4:7], 2)
        SR = int(instruction[7:10], 2)
        return f"NOT {r(DR)}, {r(SR)}"

    # JMP / RET
    if op == "1100":
        BaseR = int(instruction[7:10], 2)
        if BaseR == 7:
            return "RET"
        return f"JMP {r(BaseR)}"

    # JSR / JSRR
    if op == "0100":
        long_flag = instruction[4]
        if long_flag == "1":
            off11 = sign_extend(int(instruction[5:16], 2), 11)
            target = (pc + 1 + off11) & 0xFFFF
            return f"JSR x{target:04X}"
        else:
            BaseR = int(instruction[7:10], 2)
            return f"JSRR {r(BaseR)}"

    # RTI
    if op == "1000":
        return "RTI"

    # TRAP 和别名
    if op == "1111":
        trapvect8 = int(instruction[8:16], 2)
        for name, code in TRAPS.items():
            if code == trapvect8:
                return name
        return f"TRAP x{trapvect8:02X}"

    return "<unknown>"


def load_program_from_text(text: str) -> LC3State:
    """
    从文本加载程序：
    - 第一行：起始地址（二进制 16 位）
    - 后续每行：一条 16 位二进制机器码
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("程序为空。")

    start_bin = lines[0]
    if len(start_bin) != 16 or any(c not in "01" for c in start_bin):
        raise ValueError("起始地址必须是 16 位二进制字符串。")

    start_addr = int(start_bin, 2) & 0xFFFF

    mem = Memory()
    addr = start_addr
    for code in lines[1:]:
        if len(code) != 16 or any(c not in "01" for c in code):
            raise ValueError(f"非法机器码行：{code}")
        mem.code[addr] = code
        addr = (addr + 1) & 0xFFFF

    return LC3State(memory=mem, pc=start_addr)


def step(state: LC3State) -> None:
    """
    执行一条指令：逻辑等价于原脚本中的主循环（除了交互问答部分）。
    """
    if state.halted:
        return

    # 记录执行前的 PC（用于标识刚刚执行的机器码）
    prev_pc = state.pc

    memory = state.memory
    Registers = state.registers
    CC = state.cc
    PC = state.pc

    instruction = memory.code[PC]
    opcode = instruction[0:4]
    PC = (PC + 1) & 0xFFFF  # 默认 PC+1

    # 在修改前重置 last_change
    state.last_change = None

    if opcode == "0001":  # ADD
        DR = int(instruction[4:7], 2)
        SR1 = int(instruction[7:10], 2)
        # 记录旧值
        old_val = Registers[DR]
        if instruction[10] == "0":
            SR2 = int(instruction[13:16], 2)
            Registers[DR] = (Registers[SR1] + Registers[SR2]) & 0xFFFF
        else:
            imm5 = int(instruction[11:16], 2)
            if imm5 >= 16:
                imm5 = imm5 - 32
            Registers[DR] = (Registers[SR1] + imm5) & 0xFFFF
        # 记录寄存器写入
        state.last_change = {"type": "reg", "index": DR, "old": old_val}
        if Registers[DR] == 0:
            CC = "Z"
        elif (Registers[DR] >> 15) & 1:
            CC = "N"
        else:
            CC = "P"

    elif opcode == "0101":  # AND
        DR = int(instruction[4:7], 2)
        SR1 = int(instruction[7:10], 2)
        old_val = Registers[DR]
        if instruction[10] == "0":
            SR2 = int(instruction[13:16], 2)
            Registers[DR] = (Registers[SR1] & Registers[SR2]) & 0xFFFF
        else:
            imm5 = int(instruction[11:16], 2)
            if imm5 >= 16:
                imm5 = imm5 - 32
            Registers[DR] = (Registers[SR1] & imm5) & 0xFFFF
        state.last_change = {"type": "reg", "index": DR, "old": old_val}
        if Registers[DR] == 0:
            CC = "Z"
        elif (Registers[DR] >> 15) & 1:
            CC = "N"
        else:
            CC = "P"

    elif opcode == "1111":  # TRAP
        trapvect8 = instruction[8:16]
        if trapvect8 == "00100101":  # HALT
            state.last_change = {"type": "halt", "old": state.halted}
            state.halted = True

    elif opcode == "0000":  # BR
        n = instruction[4]
        z = instruction[5]
        p = instruction[6]
        pc_offset9 = int(instruction[7:16], 2)
        if pc_offset9 >= 256:
            pc_offset9 = pc_offset9 - 512
        if (n == "1" and CC == "N") or (z == "1" and CC == "Z") or (p == "1" and CC == "P"):
            PC = (PC + pc_offset9) & 0xFFFF

    elif opcode == "0010":  # LD
        DR = int(instruction[4:7], 2)
        old_val = Registers[DR]
        pc_offset9 = int(instruction[7:16], 2)
        if pc_offset9 >= 256:
            pc_offset9 = pc_offset9 - 512
        addr = (PC + pc_offset9) & 0xFFFF
        Registers[DR] = int(memory.code[addr], 2)
        state.last_change = {"type": "reg", "index": DR, "old": old_val}
        if Registers[DR] == 0:
            CC = "Z"
        elif (Registers[DR] >> 15) & 1:
            CC = "N"
        else:
            CC = "P"

    elif opcode == "0011":  # ST
        SR = int(instruction[4:7], 2)
        pc_offset9 = int(instruction[7:16], 2)
        if pc_offset9 >= 256:
            pc_offset9 = pc_offset9 - 512
        addr = (PC + pc_offset9) & 0xFFFF
        old_mem = memory.code[addr]
        memory.code[addr] = format(Registers[SR] & 0xFFFF, "016b")
        state.last_change = {"type": "mem", "addr": addr, "old": old_mem}

    elif opcode == "1100":  # JMP
        BaseR = int(instruction[7:10], 2)
        old_pc = PC
        PC = Registers[BaseR] & 0xFFFF
        state.last_change = {"type": "pc", "old": old_pc}

    elif opcode == "1001":  # NOT
        DR = int(instruction[4:7], 2)
        old_val = Registers[DR]
        SR = int(instruction[7:10], 2)
        Registers[DR] = (~Registers[SR]) & 0xFFFF
        state.last_change = {"type": "reg", "index": DR, "old": old_val}
        if Registers[DR] == 0:
            CC = "Z"
        elif (Registers[DR] >> 15) & 1:
            CC = "N"
        else:
            CC = "P"

    elif opcode == "0100":  # JSR
        long_flag = instruction[4]
        old_r7 = Registers[7]
        Registers[7] = PC & 0xFFFF
        if long_flag == "1":
            pc_offset11 = int(instruction[5:16], 2)
            if pc_offset11 >= 1024:
                pc_offset11 = pc_offset11 - 2048
            old_pc = PC
            PC = (PC + pc_offset11) & 0xFFFF
            state.last_change = {"type": "pc_r7", "old_pc": old_pc, "old_r7": old_r7}
        else:
            BaseR = int(instruction[7:10], 2)
            old_pc = PC
            PC = Registers[BaseR] & 0xFFFF
            state.last_change = {"type": "pc_r7", "old_pc": old_pc, "old_r7": old_r7}

    elif opcode == "0110":  # LDR
        DR = int(instruction[4:7], 2)
        old_val = Registers[DR]
        BaseR = int(instruction[7:10], 2)
        offset6 = int(instruction[10:16], 2)
        if offset6 >= 32:
            offset6 = offset6 - 64
        addr = (Registers[BaseR] + offset6) & 0xFFFF
        Registers[DR] = int(memory.code[addr], 2)
        state.last_change = {"type": "reg", "index": DR, "old": old_val}
        if Registers[DR] == 0:
            CC = "Z"
        elif (Registers[DR] >> 15) & 1:
            CC = "N"
        else:
            CC = "P"

    elif opcode == "0111":  # STR
        SR = int(instruction[4:7], 2)
        BaseR = int(instruction[7:10], 2)
        offset6 = int(instruction[10:16], 2)
        if offset6 >= 32:
            offset6 = offset6 - 64
        addr = (Registers[BaseR] + offset6) & 0xFFFF
        old_mem = memory.code[addr]
        memory.code[addr] = format(Registers[SR] & 0xFFFF, "016b")
        state.last_change = {"type": "mem", "addr": addr, "old": old_mem}

    elif opcode == "1000":  # RTI
        # 未实现，占位
        pass

    elif opcode == "1010":  # LDI
        DR = int(instruction[4:7], 2)
        old_val = Registers[DR]
        pc_offset9 = int(instruction[7:16], 2)
        if pc_offset9 >= 256:
            pc_offset9 = pc_offset9 - 512
        addr1 = (PC + pc_offset9) & 0xFFFF
        addr2 = int(memory.code[addr1], 2)
        Registers[DR] = int(memory.code[addr2], 2)
        state.last_change = {"type": "reg", "index": DR, "old": old_val}
        if Registers[DR] == 0:
            CC = "Z"
        elif (Registers[DR] >> 15) & 1:
            CC = "N"
        else:
            CC = "P"

    elif opcode == "1011":  # STI
        SR = int(instruction[4:7], 2)
        pc_offset9 = int(instruction[7:16], 2)
        if pc_offset9 >= 256:
            pc_offset9 = pc_offset9 - 512
        addr1 = (PC + pc_offset9) & 0xFFFF
        addr2 = int(memory.code[addr1], 2)
        old_mem = memory.code[addr2]
        memory.code[addr2] = format(Registers[SR] & 0xFFFF, "016b")
        state.last_change = {"type": "mem", "addr": addr2, "old": old_mem}

    elif opcode == "1110":  # LEA
        DR = int(instruction[4:7], 2)
        old_val = Registers[DR]
        pc_offset9 = int(instruction[7:16], 2)
        if pc_offset9 >= 256:
            pc_offset9 = pc_offset9 - 512
        addr = (PC + pc_offset9) & 0xFFFF
        Registers[DR] = addr
        state.last_change = {"type": "reg", "index": DR, "old": old_val}

    else:
        # 未知指令，忽略
        pass

    # 记录“刚刚执行的指令地址”
    state.last_executed_pc = prev_pc
    state.registers = Registers
    state.cc = CC
    state.pc = PC


def main() -> None:
    st.set_page_config(page_title="LC-3 模拟器可视化", layout="wide")
    st.title("LC-3 模拟器可视化")
    st.markdown(
        """
**使用说明：**
- 输入的格式为：第一行是起始地址（16 位二进制），后面每行是一条 16 位二进制机器码。
- 点击“加载程序”初始化模拟器。
- 使用“单步执行”或“运行到 HALT”逐步观察状态。
- 寄存器和内存均以十六进制展示（前缀 `x`）。
"""
    )

    if "state" not in st.session_state:
        st.session_state["state"] = None
    if "history_stack" not in st.session_state:
        # 用于多步回退：保存一个栈，每个元素是一小步执行前的快照
        st.session_state["history_stack"] = []

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("程序输入（二进制）")
        default_prog = """0011000000000000
0001000000100001
0000000000000000
1111000000100101
"""
        prog_text = st.text_area(
            "第一行：起始地址；后续：每行一条 16 位机器码：",
            value=default_prog,
            height=260,
            key="prog_input",
        )

        if st.button("加载程序"):
            try:
                st.session_state["state"] = load_program_from_text(prog_text)
                st.success("程序加载成功。")
            except Exception as e:
                st.session_state["state"] = None
                st.error(f"加载失败：{e}")

        st.markdown("---")
        st.subheader("执行 / 回退")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("单步执行"):
                state: LC3State | None = st.session_state.get("state")
                if state is None:
                    st.warning("请先加载程序。")
                elif state.halted:
                    st.info("程序已 HALT。")
                else:
                    # 记录这一步之前用于回退的信息，压入栈顶
                    history_stack = st.session_state["history_stack"]
                    history_stack.append(
                        {
                            "pc": state.pc,
                            "cc": state.cc,
                            "registers": state.registers.copy(),
                            "memory": None,  # 在真正写内存时再更新为 (addr, old_val)
                        }
                    )
                    st.session_state["history_stack"] = history_stack
                    step(state)
                    # 如果这一步修改了内存，在 history 中补充具体改动
                    change = state.last_change
                    if change and change.get("type") == "mem":
                        addr = change["addr"]
                        history_stack[-1]["memory"] = (addr, change["old"])

        with c2:
            if st.button("回退一步"):
                state: LC3State | None = st.session_state.get("state")
                history_stack = st.session_state.get("history_stack", [])
                if state is None or not history_stack:
                    st.warning("当前没有可回退的步骤。")
                else:
                    # 弹出最近的一步
                    hist = history_stack.pop()
                    st.session_state["history_stack"] = history_stack

                    # 恢复寄存器、PC、CC
                    state.registers = hist["registers"]
                    state.pc = hist["pc"]
                    state.cc = hist["cc"]
                    # 如有内存改动，恢复它
                    mem_change = hist.get("memory")
                    if mem_change is not None:
                        addr, old_val = mem_change
                        state.memory.code[addr] = old_val

    with col_right:
        st.subheader("寄存器与标志位")
        state: LC3State | None = st.session_state.get("state")
        if state is None:
            st.info("尚未加载程序。")
        else:
            # 寄存器 + 标志展示
            regs_table = {
                "寄存器": [f"R{i}" for i in range(8)] + ["PC", "CC"],
                "值 (十六进制)": [f"x{reg:04X}" for reg in state.registers]
                + [f"x{state.pc:04X}", state.cc],
            }
            st.table(regs_table)

            # 当前 / 刚刚执行 & 下一条即将执行的指令（反汇编）
            st.subheader("指令视图")
            # 当前/刚刚执行的地址：若有 last_executed_pc，用它；否则为当前 PC
            current_pc = (
                state.last_executed_pc if state.last_executed_pc is not None else state.pc
            )
            current_instr = state.memory.code[current_pc]
            with st.expander("当前/刚刚执行的指令", expanded=True):
                if current_instr == "0000000000000000":
                    st.text(f"地址 x{current_pc:04X}: 内存中该位置为空（全 0）。")
                else:
                    asm_cur = disassemble(current_instr, current_pc)
                    st.code(
                        f"地址 x{current_pc:04X}: {current_instr}    ; {asm_cur}",
                        language="text",
                    )

            # 下一条即将执行的地址：如果已经执行过至少一条，则为 state.pc；否则为 current_pc+1
            if state.last_executed_pc is not None:
                next_pc = state.pc
            else:
                next_pc = (current_pc + 1) & 0xFFFF

            next_instr = state.memory.code[next_pc]
            with st.expander("下一条即将执行的指令", expanded=True):
                if next_instr == "0000000000000000":
                    st.text(f"地址 x{next_pc:04X}: 内存中该位置为空（全 0）。")
                else:
                    asm_next = disassemble(next_instr, next_pc)
                    st.code(
                        f"地址 x{next_pc:04X}: {next_instr}    ; {asm_next}",
                        language="text",
                    )

            # 内存视图：高亮刚刚执行的指令
            st.subheader("内存视图（仅显示非零内容）")
            rows = []
            for addr, code in enumerate(state.memory.code):
                if code != "0000000000000000":
                    val = int(code, 2)
                    marker = ""
                    if state.last_executed_pc is not None and addr == state.last_executed_pc:
                        marker = "★ 刚刚执行"
                    elif addr == state.pc:
                        marker = "→ 即将执行 (PC)"
                    rows.append(
                        {
                            "地址": f"x{addr:04X}",
                            "值 (十六进制)": f"x{val:04X}",
                            "值 (二进制)": code,
                            "标记": marker,
                        }
                    )
            if rows:
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("当前内存中没有非零内容。")


if __name__ == "__main__":
    main()
