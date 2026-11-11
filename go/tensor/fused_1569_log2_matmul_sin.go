package tensor

import "fmt"

func Fused1569Log2MatmulSin(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_sin: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Sin()
	return r, nil
}
func Fused1569Log2MatmulSinName() string { return "log2_matmul_sin" }
