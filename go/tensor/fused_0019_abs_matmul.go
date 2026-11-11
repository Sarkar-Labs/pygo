package tensor

import "fmt"

func Fused0019AbsMatmul(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	return r, nil
}
func Fused0019AbsMatmulName() string { return "abs_matmul" }
