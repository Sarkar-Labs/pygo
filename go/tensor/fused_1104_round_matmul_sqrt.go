package tensor

import "fmt"

func Fused1104RoundMatmulSqrt(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("round_matmul_sqrt: nil operand")
	}
	r := x.MatMul(y)
	r = r.Round()
	r = r.Sqrt()
	return r, nil
}
func Fused1104RoundMatmulSqrtName() string { return "round_matmul_sqrt" }
