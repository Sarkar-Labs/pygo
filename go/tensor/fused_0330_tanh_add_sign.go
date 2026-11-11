package tensor

import "fmt"

func Fused0330TanhAddSign(x, y *Tensor) (*Tensor, error) {
	if x == nil || y == nil {
		return nil, fmt.Errorf("tanh_add_sign: nil operand")
	}
	r := x.Add(y)
	r = r.Tanh()
	r = r.Sign()
	return r, nil
}
func Fused0330TanhAddSignName() string { return "tanh_add_sign" }
