package tensor

import "fmt"

func Fused1099RoundMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Abs()
	return r, nil
}
func Fused1099RoundMatmulAbsName() string { return "round_matmul_abs" }
