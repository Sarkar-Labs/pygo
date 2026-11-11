package tensor

import "fmt"

func Fused1559Log2MatmulExp(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("log2_matmul_exp: nil operand")
	}
	r := x.MatMul(y)
	r = r.Log2()
	r = r.Exp()
	return r, nil
}
func Fused1559Log2MatmulExpName() string { return "log2_matmul_exp" }
