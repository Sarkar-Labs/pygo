package tensor

import "fmt"

func Fused0289TanhMatmulAbs(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_matmul_abs: nil operand")
	}
	r := x.MatMul(y)
	r = r.Tanh()
	r = r.Abs()
	return r, nil
}
func Fused0289TanhMatmulAbsName() string { return "tanh_matmul_abs" }
