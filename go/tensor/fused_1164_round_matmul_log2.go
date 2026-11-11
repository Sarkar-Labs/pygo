package tensor

import "fmt"

func Fused1164RoundMatmulLog2(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_log2: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Log2()
	return r, nil
}
func Fused1164RoundMatmulLog2Name() string { return "round_matmul_log2" }
