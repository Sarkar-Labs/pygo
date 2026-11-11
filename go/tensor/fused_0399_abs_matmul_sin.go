package tensor

import "fmt"

func Fused0399AbsMatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("abs_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Abs()
	r = r.Sin()
	return r, nil
}
func Fused0399AbsMatmulSinName() string { return "abs_matmul_sin" }
